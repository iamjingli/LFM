import numpy as np

Office = [0 for i in range(6)]
o_i = 0

CLIP_Office = [0 for ii in range(6)]
co_i = 0

OfficeHome = [0 for j in range(12)]
of_i = 0

CLIP_OfficeHome = [0 for jj in range(12)]
cof_i = 0

VisDA = [0 for k in range(1)]
v_i = 0

CLIP_VisDA = [0 for kk in range(1)]
cv_i = 0

DomainNet = [0 for v in range(6)]
d_i = 0

sc = ["result_tcl"]
set_ = ["CLDA"]
start = 0

for n in range(1):
    with open(f"{sc[n]}.txt", "r", encoding="utf-8") as f:
        m = 3
        p = 7
        ps = 9
        res = f.readlines()
        for i in res:
            if len(i) > 50 and start == 0:
                s = i.split(' ')
                if s[1] == "Office":
                    Office[o_i] += float(s[p][ps:-1]) / m * 100
                    o_i += 1
                    if o_i == 6:
                        o_i = 0

                elif s[1] == "OfficeHome":
                    OfficeHome[of_i] += float(s[p][ps:-1]) / m * 100
                    of_i += 1
                    if of_i == 12:
                        of_i = 0

                elif s[1] == "VisDA":
                    VisDA[v_i] += float(s[p][ps:-1]) / m * 100
                    v_i += 1
                    if v_i == 1:
                        v_i = 0

    with open("result.txt", "a+") as f:
        f.write(f"========================================{set_[n]}========================================\n")
        f.write("Office: \n")
        f.write("CLIP+LLM:\t")
        avg = np.mean(Office)
        for i in range(len(Office)):
            f.write(f"{Office[i]:.1f}\t")
            Office[i] = 0
        f.write(f"Avg. {avg:.1f}")
        f.write("\n\n")

        f.write("OfficeHome:\n")
        f.write("Ours:\t\t")
        avg = np.mean(OfficeHome)
        for i in range(len(OfficeHome)):
            f.write(f"{OfficeHome[i]:.1f}\t")
            OfficeHome[i] = 0
        f.write(f"Avg. {avg:.1f}")
        f.write("\n\n")

        f.write("VisDA:\n")
        f.write("Ours:\t\t")
        for i in range(len(VisDA)):
            f.write(f"{VisDA[i]:.1f}\t")
            VisDA[i] = 0
        f.write(f"\n====================================================================================\n")
        f.write("\n\n")
