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

CLIP_DomainNet = [0 for vv in range(6)]
cd_i = 0

sc = ["result_tos", "result_top", "result_tp"]
set_ = ["OSDA", "OPDA", "PDA"]

for n in range(3):
    with open(f"{sc[n]}.txt", "r", encoding="utf-8") as f:
        if n == 2:
            m = 1
            p = 7
            ps = 9
        else:
            m = 3
            p = 6
            ps = 0
        res = f.readlines()
        for i in res:
            if len(i) > 50:
                s = i.split(' ')
                if s[1] == "Office":
                    Office[o_i] += float(s[p][ps:-1]) / m * 100
                    o_i += 1
                    if o_i == 6:
                        o_i = 0

                    CLIP_Office[co_i] += float(s[-1][8:]) / m * 100
                    co_i += 1
                    if co_i == 6:
                        co_i = 0

                elif s[1] == "OfficeHome":
                    OfficeHome[of_i] += float(s[p][ps:-1]) / m * 100
                    of_i += 1
                    if of_i == 12:
                        of_i = 0

                    CLIP_OfficeHome[cof_i] += float(s[-1][8:]) / m * 100
                    cof_i += 1
                    if cof_i == 12:
                        cof_i = 0

                elif s[1] == "VisDA":
                    VisDA[v_i] += float(s[p][ps:-1]) / m * 100
                    v_i += 1
                    if v_i == 1:
                        v_i = 0

                    CLIP_VisDA[cv_i] += float(s[-1][8:]) / m * 100
                    cv_i += 1
                    if cv_i == 1:
                        cv_i = 0

                elif s[1] == "DomainNet":
                    DomainNet[d_i] += float(s[p][ps:-1]) / m * 100
                    d_i += 1
                    if d_i == 6:
                        d_i = 0

                    CLIP_DomainNet[cd_i] += float(s[-1][8:]) / m * 100
                    cd_i += 1
                    if cd_i == 6:
                        cd_i = 0

    with open("result.txt", "a+") as f:
        f.write(f"========================================{set_[n]}========================================\n")
        f.write("Office: \n")
        f.write("CLIP+LLM:\t")
        avg = np.mean(CLIP_Office)
        for i in range(len(CLIP_Office)):
            f.write(f"{CLIP_Office[i]:.1f}\t")
            CLIP_Office[i] = 0
        f.write(f"Avg. {avg:.1f}\n")
        f.write("Ours:\t\t")
        avg = np.mean(Office)
        for i in range(len(Office)):
            f.write(f"{Office[i]:.1f}\t")
            Office[i] = 0
        f.write(f"Avg. {avg:.1f}")
        f.write("\n\n")

        f.write("OfficeHome:\n")
        f.write("CLIP+LLM:\t")
        avg = np.mean(CLIP_OfficeHome)
        for i in range(len(CLIP_OfficeHome)):
            f.write(f"{CLIP_OfficeHome[i]:.1f}\t")
            CLIP_OfficeHome[i] = 0
        f.write(f"Avg. {avg:.1f}\n")
        f.write("Ours:\t\t")
        avg = np.mean(OfficeHome)
        for i in range(len(OfficeHome)):
            f.write(f"{OfficeHome[i]:.1f}\t")
            OfficeHome[i] = 0
        f.write(f"Avg. {avg:.1f}")
        f.write("\n\n")

        if set_[1] == "OPDA":
            f.write("DomainNet:\n")
            f.write("CLIP+LLM:\t")
            avg = np.mean(CLIP_DomainNet)
            for i in range(len(CLIP_DomainNet)):
                f.write(f"{CLIP_DomainNet[i]:.1f}\t")
                CLIP_DomainNet[i] = 0
            f.write(f"Avg. {avg:.1f}\n")
            f.write("Ours:\t\t")
            avg = np.mean(DomainNet)
            for i in range(len(DomainNet)):
                f.write(f"{DomainNet[i]:.1f}\t")
                DomainNet[i] = 0
            f.write(f"Avg. {avg:.1f}")
            f.write("\n\n")

        f.write("VisDA:\n")
        f.write("CLIP+LLM:\t")
        for i in range(len(CLIP_VisDA)):
            f.write(f"{CLIP_VisDA[i]:.1f}\t")
            CLIP_VisDA[i] = 0
        f.write("\n")
        f.write("Ours:\t\t")
        for i in range(len(VisDA)):
            f.write(f"{VisDA[i]:.1f}\t")
            VisDA[i] = 0
        f.write(f"\n====================================================================================\n")
        f.write("\n\n")
