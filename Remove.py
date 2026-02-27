"""
remove_outliers.py — 移除异常数据文件（共 64 个）
运行一次即可：python remove_outliers.py
"""

import os

DATA_DIR = "data"

# 从 0/ 文件夹移除（23 个文件，score > 0.5）
REMOVE_FROM_0 = [
    "PD-Ruijin_310103196109170842_tappingleft1_2020-12-15_19-44-16.txt",
    "PD-Ruijin_332623194607184349_tappingright1_2020-12-07_18-01-09.txt",
    "PD-Ruijin_310111195010160439_tappingleft1_2020-09-07_18-01-01.txt",
    "PD-Ruijin_320902195602092010_tappingright1_2020-12-02_11-00-57.txt",
    "PD-Ruijin_32062519490128442X_tappingright1_2021-01-13_14-10-14.txt",
    "PD-Ruijin_310225196303066014_tappingright1_2020-12-09_12-28-08.txt",
    "PD-Ruijin_310112197401251018_tappingleft1_2020-12-07_19-01-37.txt",
    "PD-Ruijin_330211194610240023_tappingright1_2021-01-13_14-10-14.txt",
    "PD-Ruijin_32072319540321221X_tappingright1_2020-09-09_13-04-22.txt",
    "PD-Ruijin_332623195610070021_tappingright1_2020-08-28_16-34-47.txt",
    "PD-Ruijin_310101195109121613_tappingright1_2021-02-23_18-27-24.txt",
    "PD-Ruijin_310230196003223517_tappingleft1_2020-12-09_11-30-48.txt",
    "PD-Ruijin_440301194810087818_tappingright1_2021-01-12_17-54-25.txt",
    "PD-Ruijin_342622198312163827_tappingright1_2020-12-16_13-14-26.txt",
    "PD-Ruijin_310224194512313522_tappingright1_2021-01-11_19-01-08.txt",
    "PD-Ruijin_130802197605011616_tappingleft1_2020-12-18_20-08-15.txt",
    "PD-Ruijin_310102195410111226_tappingright1_2021-02-24_14-34-56.txt",
    "PD-Ruijin_310102195410111226_tappingleft1_2021-02-24_14-34-56.txt",
    "PD-Ruijin_310101195109121613_tappingleft1_2021-02-23_18-27-24.txt",
    "PD-Ruijin_310104195811210906_tappingright1_2021-03-10_11-40-09.txt",
    "PD-Ruijin_32062519490128442X_tappingleft1_2021-01-13_14-10-14.txt",
    "PD-Ruijin_450302194610131018_tappingleft1_2020-09-22_15-44-54.txt",
    "PD-Ruijin_342425196904154923_tappingright1_2021-01-11_19-54-51.txt",
]

# 从 1/ 文件夹移除（40 个文件 + 1 个重复文件 = 41 个）
REMOVE_FROM_1 = [
    "PD-Ruijin_31010319610217324X_tappingleft1_2020-12-21_19-44-50.txt",
    "PD-Ruijin_330123197604192815_tappingleft1_2021-02-01_15-17-10.txt",
    "PD-Ruijin_330901194807110017_tappingleft1_2020-12-18_19-15-20.txt",
    "PD-Ruijin_412824197907040655_tappingright1_2020-11-04_13-59-34.txt",
    "PD-Ruijin_310105196211272426_tappingright1_2020-10-12_20-32-57.txt",
    "PD-Ruijin_350125197707182448-v2_tappingright1_2021-03-01_17-32-29.txt",
    "PD-Ruijin_513825198502120417_tappingleft1_2020-09-15_16-22-53.txt",
    "PD-Ruijin_513021197408076051_tappingright1_2020-12-07_16-25-06.txt",
    "PD-Ruijin_320622196711120053_tappingleft1_2020-12-14_15-56-45.txt",
    "PD-Ruijin_310108196605105224_tappingleft1_2020-09-28_16-27-36.txt",
    "PD-Ruijin_230104197709214512_tappingright1_2021-01-06_13-52-51.txt",
    "PD-Ruijin_310103195609033241_tappingleft1_2020-11-09_19-57-28.txt",
    "PD-Ruijin_310105196211272426-v2_tappingright1_2020-10-12_20-45-17.txt",
    "PD-Ruijin_310228195512205054_tappingleft1_2021-02-05_16-47-46.txt",
    "PD-Ruijin_360429195209032724_tappingleft1_2020-11-06_17-48-30.txt",
    "PD-Ruijin_342422197608082895_tappingleft1_2021-02-22_16-30-06.txt",
    "PD-Ruijin_320722197402137361_tappingright1_2020-12-01_13-08-31.txt",
    "PD-Ruijin_310226194508130021_tappingleft1_2020-08-31_15-12-35.txt",
    "PD-Ruijin_410926195108155242_tappingleft1_2020-11-25_14-28-36.txt",
    "PD-Ruijin_320922196703046312_tappingleft1_2020-10-19_16-29-42.txt",
    "PD-Ruijin_370102194910102915_tappingleft1_2020-11-09_19-57-28.txt",
    "PD-Ruijin_320622196109110047_tappingright1_2020-10-26_19-09-19.txt",
    "PD-Ruijin_220204194404080912-v2_tappingleft1_2020-09-30_10-23-54.txt",
    "PD-Ruijin_310105196211272426-v2_tappingleft1_2020-10-12_20-45-17.txt",
    "PD-Ruijin_310111194811181243_tappingleft1_2020-10-26_15-14-05.txt",
    "PD-Ruijin_310105195401081693_tappingright1_2020-11-06_16-54-14.txt",
    "PD-Ruijin_330901195311030341_tappingleft1_2020-11-11_13-50-05.txt",
    "PD-Ruijin_330822195710150013_tappingright1_2020-10-23_15-21-17.txt",
    "PD-Ruijin_321083196610212734_tappingleft1_2020-10-27_19-46-49.txt",
    "PD-Ruijin_37010219531022294X_tappingright1_2020-11-09_17-48-09.txt",
    "PD-Ruijin_330521196210074618_tappingleft1_2020-12-16_14-31-21.txt",
    "PD-Ruijin_320521195709100081-v2_tappingleft1_2020-11-04_15-07-37.txt",
    "PD-Ruijin_310109194909014820-v2_tappingleft1_2020-09-23_15-23-53.txt",
    "PD-Ruijin_320521194401273229_tappingright1_2020-09-02_12-10-11.txt",
    "PD-Ruijin_310110197203093255_tappingleft1_2020-12-14_15-56-45.txt",
    "PD-Ruijin_330328194010184710_tappingright1_2020-11-25_14-28-36.txt",
    "PD-Ruijin_310110195501025446_tappingright1_2021-02-03_14-00-42.txt",
    "PD-Ruijin_320521195709100081-v2_tappingright1_2020-11-04_15-07-37.txt",
    "PD-Ruijin_310108195011201628_tappingleft1_2020-09-02_14-59-14.txt",
    "PD-Ruijin_220124195008234240_tappingright1_2020-10-28_11-43-43.txt",
    "PD-Ruijin_650300195302055921_tappingright1_2021-03-03_13-19-08 - 副本.txt",
]


def main():
    removed, not_found = 0, 0

    for fname in REMOVE_FROM_0:
        path = os.path.join(DATA_DIR, "0", fname)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
        else:
            print(f"  [NOT FOUND] {path}")
            not_found += 1

    for fname in REMOVE_FROM_1:
        path = os.path.join(DATA_DIR, "1", fname)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
        else:
            print(f"  [NOT FOUND] {path}")
            not_found += 1

    # Print remaining file counts
    for folder in ["0", "1"]:
        p = os.path.join(DATA_DIR, folder)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if f.endswith('.txt')])
            print(f"  Folder {folder}: {n} files remaining")

    print(f"\nDone: removed {removed}, not found {not_found}")


if __name__ == "__main__":
    main()