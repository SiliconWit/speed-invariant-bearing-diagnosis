"""Download the CWRU and JNU recordings and check them against the manifest.

    python3 data/fetch_data.py            # download what is missing, then check all
    python3 data/fetch_data.py --check    # check the files already present, download nothing

Files land in data/cwru and data/jnu next to this script (or under --dest).  Each file
is compared against the MD5 checksum of the copy used for the reported results; the
same checksums are listed in data/MANIFEST.md.  Standard library only.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

CWRU_URL = "https://engineering.case.edu/sites/default/files/{name}"
# The mirror is pinned to a commit, so the files cannot change under the same URL.
JNU_COMMIT = "75b33611b51649d1da8ff5999397899420753e5b"
JNU_URL = ("https://raw.githubusercontent.com/ClarkGableWang/JNU-Bearing-Dataset/"
           + JNU_COMMIT + "/{name}")

# (subdirectory, file name) -> MD5
MD5 = {
    ("cwru", "97.mat"): "ee410e7243aefcd8b7120876556464e7",
    ("cwru", "98.mat"): "3983dcddead0d4b910ee1e8d3da164b9",
    ("cwru", "99.mat"): "edc080a0b4fbc0c2ec8b7f1403372b73",
    ("cwru", "100.mat"): "cc57a511b95785c805055d99281ea2bb",
    ("cwru", "105.mat"): "f14821b40412f33018279198da7f9cdc",
    ("cwru", "106.mat"): "4abe03d02739813eebc879b20b591968",
    ("cwru", "107.mat"): "2bf7a32b6b59aa379d5a4a3c345418f5",
    ("cwru", "108.mat"): "ccec995fde6f8865632972804dadfc55",
    ("cwru", "118.mat"): "ff0f20588edc64140ae33888fcf1114a",
    ("cwru", "119.mat"): "ba784233ebbe424e31bb35c5523413cd",
    ("cwru", "120.mat"): "89fbf0d771512d848972fd574d3923e8",
    ("cwru", "121.mat"): "7715250229a6d47910d12be7e1970e70",
    ("cwru", "130.mat"): "e45f73f65cc3de4482d28d758503f2c1",
    ("cwru", "131.mat"): "8053f13c15bb8891dd7586466e72fb34",
    ("cwru", "132.mat"): "cf1a7ab3e14564be490c7c27bfafed78",
    ("cwru", "133.mat"): "796a8db507869acf5e0b880d0925f693",
    ("jnu", "n600_3_2.csv"): "0e9e08bc1ded3d93ee3f5b9cb8b2613c",
    ("jnu", "n800_3_2.csv"): "519b0bc94d7c9fe996099a5389b3488d",
    ("jnu", "n1000_3_2.csv"): "2dd117c214b0351babdb9ceca90efd3d",
    ("jnu", "ib600_2.csv"): "1d49a9e027f15a9bfce987a9207615d2",
    ("jnu", "ib800_2.csv"): "72da039339c2c46d2b832d6be12e0617",
    ("jnu", "ib1000_2.csv"): "94f8aa807c5ee59462b230c83322404a",
    ("jnu", "tb600_2.csv"): "7fa40a7f5baef74d7b5469d5011d1a8b",
    ("jnu", "tb800_2.csv"): "08568291b6a7973c76549a98d411d93d",
    ("jnu", "tb1000_2.csv"): "e6df120ff66e43b54adbe017a9bc92d8",
    ("jnu", "ob600_2.csv"): "dd11ccff8089bb6a8fa67ebdf4e03341",
    ("jnu", "ob800_2.csv"): "b770eb7461b14ed031625bca1a8a6c88",
    ("jnu", "ob1000_2.csv"): "2f4594cbc1092525dc7212ec40ba1050",
}

URL = {"cwru": CWRU_URL, "jnu": JNU_URL}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, tries=4):
    tmp = path + ".part"
    for k in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fetch_data.py"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            os.replace(tmp, path)
            return True
        except OSError as e:
            print(f"    attempt {k} failed: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(2 * k)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="only check files already present; download nothing")
    ap.add_argument("--dest", default=HERE,
                    help="directory that receives cwru/ and jnu/ (default: data/)")
    args = ap.parse_args()

    missing, failed, bad = [], [], []
    for (sub, name), want in MD5.items():
        folder = os.path.join(args.dest, sub)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            if args.check:
                missing.append(f"{sub}/{name}")
                continue
            url = URL[sub].format(name=name)
            print(f"  downloading {sub}/{name}")
            if not download(url, path):
                failed.append((f"{sub}/{name}", url))
                continue
        got = md5(path)
        if got != want:
            bad.append((f"{sub}/{name}", got, want))
        else:
            print(f"  ok  {sub}/{name}")

    print()
    if missing:
        print(f"{len(missing)} file(s) not present; run without --check to download them.")
    if failed:
        print("Could not download:")
        for f, url in failed:
            print(f"  {f}  from {url}")
        print("The source may be down or may have moved.  Try again later, or download the")
        print("files by hand from the pages listed in data/MANIFEST.md into the same folders.")
    if bad:
        print("Checksum mismatch:")
        for f, got, want in bad:
            print(f"  {f}  got {got}, expected {want}")
        print("These files differ from the copies used for the reported results, so the")
        print("results may not reproduce exactly.  Delete them and run this script again;")
        print("if the mismatch persists, the source has changed the files.")
    if missing or failed or bad:
        sys.exit(1)
    print(f"All {len(MD5)} files present and verified.")


if __name__ == "__main__":
    main()
