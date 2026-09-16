#!/usr/bin/env python3
# fugleramme-texas-build/build.py
# Build BirdNET-v2.4-compatible Texas artwork assets for Fugleramme.

from __future__ import annotations

import csv, io, json, re, shutil, sys, time, unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import fuzz, process as rfprocess
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent
UP = ROOT / "fugleramme-upstream"
OUT = ROOT / "output"
PATCH = OUT / "patch"
READY = OUT / "complete-ready"
REVIEW = OUT / "review"
REPORT = OUT / "reports"
SOURCES = OUT / "sources"
SPECIES = ROOT / "texas_species.txt"
COMMONS = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "Fugleramme-Texas-asset-builder/1.0 (GitHub Actions; open-source artwork curation)"}
PAPER = (0xF0, 0xEC, 0xE5)
CAP = 1200

ALIASES = {
    "mccown's longspur": "Thick-billed Longspur",
    "graylag goose": "Greylag Goose",
    "blue-throated mountain-gem": "Blue-throated Hummingbird",
    "mew gull": "Common Gull",
    "thayer's gull": "Iceland Gull",
    "cattle egret": "Western Cattle Egret",
}
KNOWN_SOURCES = {
    "audubon": ("audubon", "birds of america"),
    "gould": ("john gould", "gouldbirds"),
    "dresser": ("dresser", "history of the birds of europe"),
    "keulemans": ("keulemans", "onze vogels"),
    "vonwright": ("von wright", "svenska faglar", "svenska fåglar"),
    "morris": ("beverley r. morris", "british game birds and wildfowl"),
    "dorbigny": ("d'orbigny", "d’orbigny", "dictionnaire universel"),
    "desmurs": ("des murs", "iconographie ornithologique"),
    "bree": ("charles robert bree",),
    "whitaker": ("whitaker", "birds of tunisia"),
}
HIST = (
    "audubon", "birds of america", "illustration", "bird illustration", "plate",
    "gould", "keulemans", "dresser", "von wright", "svenska faglar",
    "ornitholog", "biodiversity heritage library", "lithograph", "engraving",
    "hand coloured", "hand colored", "watercolor", "watercolour", "des murs",
    "d orbigny", "whitaker", "morris",
)
NEG = ("range map", "distribution map", "sonogram", "spectrogram", "logo", "photograph", "flickr")

@dataclass
class Taxon:
    requested: str
    common: str
    legacy_sci: str
    sci: str
    key: str
    match: str
    score: float

@dataclass
class Candidate:
    title: str
    page: str
    direct: str
    thumb: str
    license: str
    artist: str
    credit: str
    source_key: str
    score: float


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def shape(s: str) -> str:
    return s.strip().lower().replace(" ", "-")


def html(s: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())


def meta(m: dict, key: str) -> str:
    v = m.get(key, "")
    return html(str(v.get("value", ""))) if isinstance(v, dict) else html(str(v or ""))


def load_taxonomy() -> tuple[list[tuple[str,str,str,str]], dict[str,str]]:
    raw_alias = json.loads((UP / "assets/birdnet_aliases.json").read_text())
    amap = {shape(k): v.strip() for k,v in raw_alias.items() if isinstance(k,str) and isinstance(v,str)}
    rows = []
    for line in (UP / "assets/birdnet_labels_v2.4.txt").read_text().splitlines():
        if "_" not in line: continue
        sci, common = map(str.strip, line.split("_",1))
        current = amap.get(shape(sci), sci)
        rows.append((common, sci, current, shape(current)))
    return rows, amap


def resolve(names: list[str], rows: list[tuple[str,str,str,str]]) -> tuple[list[Taxon], list[dict]]:
    by_common: dict[str, list[tuple[str,str,str,str]]] = {}
    choices = []
    for row in rows:
        n = norm(row[0]); by_common.setdefault(n, []).append(row); choices.append(n)
    out, miss = [], []
    for requested in names:
        q = ALIASES.get(requested.lower(), requested)
        n = norm(q); row = by_common.get(n, [None])[0]; method = "exact"; score = 100.0
        if row is None:
            hit = rfprocess.extractOne(n, choices, scorer=fuzz.ratio)
            if hit and hit[1] >= 94:
                row = by_common[hit[0]][0]; method = "fuzzy"; score = float(hit[1])
        if row is None:
            miss.append({"requested": requested, "reason": "not in BirdNET v2.4"}); continue
        common, legacy, current, key = row
        out.append(Taxon(requested, common, legacy, current, key, method, score))
    return out, miss


def commons(query: str, s: requests.Session) -> list[dict]:
    p = {"action":"query","generator":"search","gsrsearch":query,"gsrnamespace":6,"gsrlimit":15,
         "prop":"imageinfo","iiprop":"url|size|mime|extmetadata","iiurlwidth":1800,
         "format":"json","formatversion":2}
    try:
        r=s.get(COMMONS,params=p,timeout=30); r.raise_for_status(); time.sleep(.08)
        return r.json().get("query",{}).get("pages",[]) or []
    except Exception as e:
        print("COMMONS ERROR", query, e, flush=True); return []


def public_domain(license_name: str, h: str) -> bool:
    x = norm(license_name + " " + h)
    return any(t in x for t in ("public domain","cc0","cc zero","pd old","pd art","no known copyright restrictions"))


def candidate(page: dict, t: Taxon, bonus: float) -> Candidate | None:
    ii = (page.get("imageinfo") or [None])[0]
    if not isinstance(ii,dict): return None
    em = ii.get("extmetadata") or {}
    title=str(page.get("title","")); lic=meta(em,"LicenseShortName"); artist=meta(em,"Artist"); credit=meta(em,"Credit")
    desc=meta(em,"ImageDescription"); cats=meta(em,"Categories"); src=meta(em,"Source"); obj=meta(em,"ObjectName")
    h=" ".join((title,lic,artist,credit,desc,cats,src,obj)); hn=norm(h)
    if not public_domain(lic,h): return None
    if not str(ii.get("mime","image/")).startswith("image/"): return None
    direct=str(ii.get("url", "")); thumb=str(ii.get("thumburl", "")) or direct
    if not direct or not thumb: return None
    sci = norm(t.sci); req=norm(t.requested); common=norm(t.common)
    species_match = (sci and sci in hn) or (req and req in hn) or (common and common in hn)
    historical = any(norm(x) in hn for x in HIST)
    if not species_match or not historical: return None
    score=bonus
    if "audubon" in hn or "birds of america" in hn: score += 160
    if sci in hn: score += 100
    if req in hn or common in hn: score += 80
    if "illustration" in hn or "plate" in hn: score += 40
    if any(norm(x) in hn for x in NEG): score -= 220
    w=int(ii.get("width") or 0); hgt=int(ii.get("height") or 0)
    if max(w,hgt) >= 1200: score += 20
    if max(w,hgt) < 500: score -= 100
    sk="commons-pd"
    for key, pats in KNOWN_SOURCES.items():
        if any(norm(p) in hn for p in pats): sk=key; break
    pageurl="https://commons.wikimedia.org/wiki/" + quote(title.replace(" ","_"),safe=":()_',-")
    return Candidate(title,pageurl,direct,thumb,lic,artist,credit,sk,score)


def find_source(t: Taxon, s: requests.Session) -> Candidate | None:
    qs=[(f'"{t.sci}" Audubon',140),(f'"{t.requested}" Audubon',130),
        (f'"{t.sci}" bird illustration',100),(f'"{t.requested}" bird illustration',90),
        (f'"{t.sci}" Gould',75),(f'"{t.sci}" Keulemans',70)]
    found=[]; seen=set()
    for q,b in qs:
        for p in commons(q,s):
            title=str(p.get("title",""))
            if title in seen: continue
            seen.add(title); c=candidate(p,t,b)
            if c: found.append(c)
        if found and max(c.score for c in found) >= 430: break
    return max(found,key=lambda x:x.score) if found else None


def download(c: Candidate, key: str, s: requests.Session) -> Path:
    SOURCES.mkdir(parents=True,exist_ok=True); ext=".jpg"
    u=c.thumb
    for i in range(3):
        try:
            r=s.get(u,timeout=60); r.raise_for_status(); ct=r.headers.get("content-type","")
            if "png" in ct: ext=".png"
            elif "webp" in ct: ext=".webp"
            p=SOURCES/(key+ext); p.write_bytes(r.content); return p
        except Exception:
            if i==2: raise
            time.sleep(1+i)
    raise RuntimeError("download failed")


def cleanup(img: Image.Image) -> Image.Image:
    a=np.array(img.convert("RGBA")); alpha=a[...,3]; mask=(alpha>24).astype(np.uint8)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    if n<=1: return img.convert("RGBA")
    areas=[(int(stats[i,cv2.CC_STAT_AREA]),i) for i in range(1,n)]; areas.sort(reverse=True); largest=areas[0][0]
    keep=np.zeros_like(mask); H,W=mask.shape
    for area,i in areas:
        cx,cy=cent[i]; central=.04*W<=cx<=.96*W and .03*H<=cy<=.95*H
        if area>=max(18,int(largest*.008),int(H*W*.00025)) and (central or area>H*W*.008): keep[lab==i]=1
    keep=cv2.dilate(keep,np.ones((3,3),np.uint8),iterations=1); a[...,3]=np.where(keep,alpha,0)
    return Image.fromarray(a,"RGBA")


def paper_fallback(base: Image.Image) -> Image.Image:
    rgb=np.array(base.convert("RGB")); H,W,_=rgb.shape
    b=max(4,min(H,W)//35); border=np.concatenate((rgb[:b].reshape(-1,3),rgb[-b:].reshape(-1,3),rgb[:,:b].reshape(-1,3),rgb[:,-b:].reshape(-1,3)))
    bg=np.median(border,axis=0).astype(np.uint8).reshape(1,1,3)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32); bl=cv2.cvtColor(bg,cv2.COLOR_RGB2LAB).astype(np.float32)[0,0]
    dist=np.linalg.norm(lab-bl,axis=2); fg=(dist>18).astype(np.uint8)*255
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    return cleanup(Image.fromarray(np.dstack((rgb,fg)).astype(np.uint8),"RGBA"))


def crop_alpha(img: Image.Image) -> Image.Image:
    a=np.array(img)[...,3]; ys,xs=np.where(a>2)
    if len(xs)==0: raise RuntimeError("empty alpha")
    return img.crop((int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1))


def resize(img: Image.Image, cap: int) -> Image.Image:
    if max(img.size)<=cap: return img
    k=cap/max(img.size); return img.resize((max(1,round(img.width*k)),max(1,round(img.height*k))),Image.Resampling.LANCZOS)


def cutout(path: Path, model) -> tuple[Image.Image,list[str]]:
    flags=[]
    with Image.open(path) as im: base=resize(im.convert("RGBA"),1600)
    try:
        x=remove(base,session=model,post_process_mask=True)
        if not isinstance(x,Image.Image): x=Image.open(io.BytesIO(x))
        x=cleanup(x.convert("RGBA"))
    except Exception as e:
        flags.append("rembg_failed:"+type(e).__name__); x=paper_fallback(base)
    occ=float((np.array(x)[...,3]>24).mean())
    if occ>.74:
        y=paper_fallback(base); yo=float((np.array(y)[...,3]>24).mean())
        if .003<yo<occ: x=y; flags.append("paper_fallback")
    if occ<.003: flags.append("low_foreground")
    x=crop_alpha(x); a=np.array(x)[...,3]; mask=(a>2).astype(np.uint8)
    radius=max(8,round(18*max(x.size)/2000)); k=2*radius+1
    dil=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),iterations=1)
    arr=np.zeros((x.height,x.width,4),np.uint8); arr[...,:3]=PAPER; arr[...,3]=dil*255
    out=Image.alpha_composite(Image.fromarray(arr,"RGBA"),x); out=resize(crop_alpha(out),CAP)
    aa=np.array(out)[...,3]
    if aa.min()==255: flags.append("no_transparency")
    return out,flags


def save(img: Image.Image,p: Path):
    p.parent.mkdir(parents=True,exist_ok=True); img.save(p,"WEBP",quality=90,alpha_quality=100,method=6)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in rows for k in r}) if rows else ["status"]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def contact_sheets(items: list[tuple[str,Path]], dest: Path):
    dest.mkdir(parents=True,exist_ok=True); cols,rows=5,6; cw,ch=260,220
    for page in range((len(items)+cols*rows-1)//(cols*rows)):
        subset=items[page*cols*rows:(page+1)*cols*rows]; sheet=Image.new("RGB",(cols*cw,rows*ch),(240,236,229)); d=ImageDraw.Draw(sheet)
        for j,(name,p) in enumerate(subset):
            r,c=divmod(j,cols)
            try:
                with Image.open(p) as im: im=im.convert("RGBA"); im.thumbnail((220,175),Image.Resampling.LANCZOS)
                tile=Image.new("RGBA",(cw,ch),(240,236,229,255)); tile.alpha_composite(im,((cw-im.width)//2,4)); sheet.paste(tile.convert("RGB"),(c*cw,r*ch))
                d.text((c*cw+7,r*ch+183),name[:35],fill=(30,30,30))
            except Exception: pass
        sheet.save(dest/f"contact-{page+1:02d}.jpg",quality=88)


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for p in (PATCH,READY,REVIEW,REPORT): p.mkdir(parents=True,exist_ok=True)
    names=[x.strip() for x in SPECIES.read_text().splitlines() if x.strip()]
    rows,_=load_taxonomy(); taxa,unmatched=resolve(names,rows); rank={n:i+1 for i,n in enumerate(names)}
    classic=UP/"assets/artwork/classic"; birds=classic/"birds"
    manifest=json.loads((classic/"manifest.json").read_text()); attr=(classic/"ATTRIBUTION.md").read_text()
    patch_birds=PATCH/"assets/artwork/classic/birds"; patch_birds.mkdir(parents=True,exist_ok=True)
    s=requests.Session(); s.headers.update(HEADERS); model=None; ledger=[]; qa=[]; visual=[]
    existing=generated=failed=0
    for i,t in enumerate(sorted(taxa,key=lambda z:rank[z.requested]),1):
        print(f"[{i}/{len(taxa)}] {t.requested} -> {t.sci}",flush=True)
        ex=[]
        for ext in (".webp",".png"):
            p=birds/(t.key+ext)
            if p.exists(): ex=[p]; break
        if ex:
            existing+=1; p=ex[0]; shutil.copy2(p,READY/p.name); e=manifest.get("birds/"+p.name,{})
            ledger.append({"rank":rank[t.requested],"common":t.requested,"scientific":t.sci,"key":t.key,"status":"existing","asset":p.name,"source":e.get("source",""),"url":e.get("url","")}); continue
        c=find_source(t,s)
        if not c:
            failed+=1; ledger.append({"rank":rank[t.requested],"common":t.requested,"scientific":t.sci,"key":t.key,"status":"missing-source"}); continue
        try:
            src=download(c,t.key,s)
            if model is None: model=new_session("u2netp")
            img,flags=cutout(src,model); name=t.key+".webp"; dst=READY/name; save(img,dst); save(img,patch_birds/name)
            manifest["birds/"+name]={"source":c.source_key,"url":c.page}; generated+=1; visual.append((name,dst))
            ledger.append({"rank":rank[t.requested],"common":t.requested,"scientific":t.sci,"key":t.key,"status":"generated","asset":name,"source":c.source_key,"url":c.page,"license":c.license,"artist":c.artist,"credit":c.credit,"candidate":c.title,"candidate_score":round(c.score,1)})
            qa.append({"rank":rank[t.requested],"asset":name,"flags":";".join(flags),"width":img.width,"height":img.height,"bytes":dst.stat().st_size})
        except Exception as e:
            failed+=1; ledger.append({"rank":rank[t.requested],"common":t.requested,"scientific":t.sci,"key":t.key,"status":"processing-failed","url":c.page,"error":repr(e)})
    pc=PATCH/"assets/artwork/classic"; pc.mkdir(parents=True,exist_ok=True)
    (pc/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    if "Commons public-domain (individual files)" not in attr:
        attr=attr.rstrip()+"\n\n**Commons public-domain (individual files)** - historical bird illustrations selected from Wikimedia Commons only when file metadata explicitly marks the work public domain, CC0, PD-Art, PD-old, or no-known-copyright-restrictions. Exact file-page provenance is recorded in `manifest.json`. Manifest key: `commons-pd`.\n"
    (pc/"ATTRIBUTION.md").write_text(attr)
    for u in unmatched: u["rank"]=rank.get(u["requested"],"")
    write_csv(REPORT/"source_ledger.csv",ledger); write_csv(REPORT/"qa.csv",qa); write_csv(REPORT/"taxonomy_unmatched.csv",unmatched)
    summary={"requested":len(names),"birdnet_resolved":len(taxa),"taxonomy_unmatched":len(unmatched),"existing_upstream":existing,"generated_new":generated,"missing_or_failed":failed,"ready_total":existing+generated}
    (REPORT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); contact_sheets(visual,REPORT/"contact-sheets")
    (OUT/"README.md").write_text("# Fugleramme Texas artwork build\n\n```json\n"+json.dumps(summary,indent=2)+"\n```\n\n`patch/` contains only new assets plus replacement manifest/attribution files. `reports/` contains source provenance and QA.\n")
    shutil.copy2(SPECIES,OUT/"texas_species.txt")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
