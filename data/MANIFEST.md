# Data manifest

The recordings are not part of this repository. `python3 data/fetch_data.py` downloads
them into `data/cwru` and `data/jnu` and checks every file against the MD5 checksums below,
which are those of the copies used for every reported result. `data/cache` holds
extracted features; the code rebuilds it on demand.

## CWRU bearing data

- Source: Case Western Reserve University Bearing Data Center,
  https://engineering.case.edu/bearingdatacenter
- Download URL pattern: `https://engineering.case.edu/sites/default/files/<n>.mat`
- Licence: not stated by the source.
- Files: normal baseline (97 to 100) and 12 kHz drive-end faults of 0.007 inch at the
  inner race (105 to 108), ball (118 to 121) and outer race at 6 o'clock (130 to 133),
  motor loads 0 to 3 hp. The code reads the drive-end accelerometer channel.
- Cite as: Case Western Reserve University Bearing Data Center, "Bearing data center,"
  https://engineering.case.edu/bearingdatacenter. For a description and benchmark of the
  data: W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the
  Case Western Reserve University data: a benchmark study," Mechanical Systems and Signal
  Processing, vol. 64-65, pp. 100-131, 2015, https://doi.org/10.1016/j.ymssp.2015.04.021.

| File | Bytes | MD5 |
|---|---:|---|
| cwru/97.mat | 3903344 | ee410e7243aefcd8b7120876556464e7 |
| cwru/98.mat | 7742720 | 3983dcddead0d4b910ee1e8d3da164b9 |
| cwru/99.mat | 15503928 | edc080a0b4fbc0c2ec8b7f1403372b73 |
| cwru/100.mat | 7770624 | cc57a511b95785c805055d99281ea2bb |
| cwru/105.mat | 2910768 | f14821b40412f33018279198da7f9cdc |
| cwru/106.mat | 2928192 | 4abe03d02739813eebc879b20b591968 |
| cwru/107.mat | 2931672 | 2bf7a32b6b59aa379d5a4a3c345418f5 |
| cwru/108.mat | 2950416 | ccec995fde6f8865632972804dadfc55 |
| cwru/118.mat | 2942112 | ff0f20588edc64140ae33888fcf1114a |
| cwru/119.mat | 2914248 | ba784233ebbe424e31bb35c5523413cd |
| cwru/120.mat | 2917752 | 89fbf0d771512d848972fd574d3923e8 |
| cwru/121.mat | 2917752 | 7715250229a6d47910d12be7e1970e70 |
| cwru/130.mat | 2928192 | e45f73f65cc3de4482d28d758503f2c1 |
| cwru/131.mat | 2938632 | 8053f13c15bb8891dd7586466e72fb34 |
| cwru/132.mat | 2914248 | cf1a7ab3e14564be490c7c27bfafed78 |
| cwru/133.mat | 2942112 | 796a8db507869acf5e0b880d0925f693 |

## JNU bearing data

- Source: Jiangnan University bearing fault dataset, public mirror with description,
  https://github.com/ClarkGableWang/JNU-Bearing-Dataset
- Download URL pattern, pinned to commit `75b33611b51649d1da8ff5999397899420753e5b`:
  `https://raw.githubusercontent.com/ClarkGableWang/JNU-Bearing-Dataset/75b33611b51649d1da8ff5999397899420753e5b/<file>`
- Licence: not stated by the source (the mirror has no licence file).
- Files: normal (n), inner race (ib), rolling element (tb) and outer race (ob) at 600, 800
  and 1000 rpm, sampled at 50 kHz.
- Cite as: Jiangnan University bearing fault dataset, public mirror with description,
  https://github.com/ClarkGableWang/JNU-Bearing-Dataset. The dataset is also described in
  Z. Zhao, T. Li, J. Wu, C. Sun, S. Wang, R. Yan and X. Chen, "Deep learning algorithms
  for rotating machinery intelligent diagnosis: an open source benchmark study," ISA
  Transactions, vol. 107, pp. 224-255, 2020, https://doi.org/10.1016/j.isatra.2020.08.010.

| File | Bytes | MD5 |
|---|---:|---|
| jnu/n600_3_2.csv | 13619397 | 0e9e08bc1ded3d93ee3f5b9cb8b2613c |
| jnu/n800_3_2.csv | 12914313 | 519b0bc94d7c9fe996099a5389b3488d |
| jnu/n1000_3_2.csv | 12754576 | 2dd117c214b0351babdb9ceca90efd3d |
| jnu/ib600_2.csv | 4360380 | 1d49a9e027f15a9bfce987a9207615d2 |
| jnu/ib800_2.csv | 4385354 | 72da039339c2c46d2b832d6be12e0617 |
| jnu/ib1000_2.csv | 4329871 | 94f8aa807c5ee59462b230c83322404a |
| jnu/tb600_2.csv | 4506334 | 7fa40a7f5baef74d7b5469d5011d1a8b |
| jnu/tb800_2.csv | 4437160 | 08568291b6a7973c76549a98d411d93d |
| jnu/tb1000_2.csv | 4225755 | e6df120ff66e43b54adbe017a9bc92d8 |
| jnu/ob600_2.csv | 4493623 | dd11ccff8089bb6a8fa67ebdf4e03341 |
| jnu/ob800_2.csv | 4300587 | b770eb7461b14ed031625bca1a8a6c88 |
| jnu/ob1000_2.csv | 4257863 | 2f4594cbc1092525dc7212ec40ba1050 |

## If a checksum does not match

The file differs from the copy used for the reported results. Delete it and run the
script again; if it still differs, the source has changed the file and results computed
from it may not reproduce exactly.
