"""Verbatim frozen section 6.1 decision entry."""
import math

FIELDS = {
    "A": ("A_vector",),
    "B": ("ci_R_cover", "ci_gain", "ci_width",
          "any_N_R_width_infinite", "ci_width_status"),
    "H3": ("R_IS_infinite", "N_IS_infinite", "ci_IS_diff", "ci_IS_status"),
    "C": ("ci_group_cover", "ci_group_gain"),
}
UNITS = {
    "A": ["%", "%", "pp", "MPa", "MPa", "MPa"],
    "B": {"ci_R_cover": "%", "ci_gain": "pp", "ci_width": "MPa"},
    "H3": {"ci_IS_diff": "MPa"},
    "C": {"ci_group_cover": "pp", "ci_group_gain": "pp"},
}
COMMON = ("delivered", "object_audit_pass", "locked_environment_match",
          "exact_environment", "units")

def finite_number(x):
    return type(x) in (int, float) and math.isfinite(x)

def contract_gate(kind, r):
    if type(kind) is not str or kind not in FIELDS or type(r) is not dict:
        return "U_AUDIT"
    if any(f not in r or r[f] is None for f in COMMON):
        return "U_DELIVERY"
    if any(type(r[f]) is not bool for f in COMMON[:4]):
        return "U_AUDIT"
    if not r["delivered"]:
        return "U_DELIVERY"
    if any(f not in r for f in FIELDS[kind]):
        return "U_DELIVERY"
    if kind == "A":
        v = r["A_vector"]
        if v is None: return "U_DELIVERY"
        if type(v) not in (list, tuple): return "U_AUDIT"
        if len(v) != 6: return "U_DELIVERY"
        if any(x is None for x in v): return "U_DELIVERY"
        if not all(finite_number(x) for x in v): return "U_AUDIT"
        if not (0 <= v[0] <= 100 and 0 <= v[1] <= 100
                and -100 <= v[2] <= 100): return "U_AUDIT"
        ci_names, allow_none = (), False
    else:
        flags = {"B": ("any_N_R_width_infinite",),
                 "H3": ("R_IS_infinite", "N_IS_infinite"), "C": ()}[kind]
        if any(r[f] is None for f in flags): return "U_DELIVERY"
        if any(type(r[f]) is not bool for f in flags): return "U_AUDIT"
        allow_none = any(r[f] for f in flags)
        ci_names = {"B": ("ci_R_cover", "ci_gain", "ci_width"),
                    "H3": ("ci_IS_diff",),
                    "C": ("ci_group_cover", "ci_group_gain")}[kind]
        optional = {"B": "ci_width", "H3": "ci_IS_diff"}.get(kind)
        # None 只允许相应无穷比较，覆盖 CI 永远必需。
        if any(r[f] is None and not (f == optional and allow_none)
               for f in ci_names): return "U_DELIVERY"
        for f in ci_names:
            v = r[f]
            if v is None: continue
            if type(v) not in (list, tuple): return "U_AUDIT"
            if len(v) != 2: return "U_DELIVERY"
            if any(x is None for x in v): return "U_DELIVERY"
        for f in ci_names:
            v = r[f]
            if v is None: continue
            if not all(finite_number(x) for x in v) or v[0] > v[1]:
                return "U_AUDIT"
            if f == "ci_R_cover" and not (0 <= v[0] <= v[1] <= 100):
                return "U_AUDIT"
            if f in ("ci_gain", "ci_group_cover") and not (-100 <= v[0] <= v[1] <= 100):
                return "U_AUDIT"
            if f == "ci_group_gain" and not (-200 <= v[0] <= v[1] <= 200):
                return "U_AUDIT"
        if optional:
            status_field = "ci_width_status" if kind == "B" else "ci_IS_status"
            if r[status_field] is None: return "U_DELIVERY"
            expected = "skipped_legal_infinity" if allow_none else "finite"
            if r[status_field] != expected or ((r[optional] is None) != allow_none):
                return "U_AUDIT"
        if allow_none:
            if "infinity_detail" not in r or r["infinity_detail"] is None:
                return "U_DELIVERY"
            detail = r["infinity_detail"]
            if type(detail) is not dict: return "U_AUDIT"
            expected_seeds = list(range(100, 120))
            for method in ("N", "R"):
                if method not in detail or detail[method] is None: return "U_DELIVERY"
                z = detail[method]
                if type(z) is not dict: return "U_AUDIT"
                required = ("W_infinite_seed_ids", "W_infinite_seed_count",
                            "IS_infinite_seed_ids", "IS_infinite_seed_count", "rows")
                if any(f not in z or z[f] is None for f in required):
                    return "U_DELIVERY"
                for metric in ("W", "IS"):
                    ids, count = z[metric+"_infinite_seed_ids"], z[metric+"_infinite_seed_count"]
                    if type(ids) is not list or type(count) is not int: return "U_AUDIT"
                    if any(type(i) is not int or i not in expected_seeds for i in ids):
                        return "U_AUDIT"
                    if ids != sorted(set(ids)) or count != len(ids): return "U_AUDIT"
                rows = z["rows"]
                if type(rows) is not list: return "U_AUDIT"
                if len(rows) != 20: return "U_DELIVERY"
                for seed, row in zip(expected_seeds, rows):
                    if type(row) is not dict: return "U_AUDIT"
                    keys = ("seed", "test_n", "empty_rows", "near_zero_empty_rows")
                    if any(f not in row or row[f] is None for f in keys): return "U_DELIVERY"
                    if any(type(row[f]) is not int for f in keys): return "U_AUDIT"
                    if (row["seed"] != seed or row["test_n"] != 252 or
                        not (0 <= row["near_zero_empty_rows"] <= row["empty_rows"] <= 252)):
                        return "U_AUDIT"
                empty_ids = [row["seed"] for row in rows if row["empty_rows"] > 0]
                # N/R 的有限走廊下，无穷主宽度/评分只来自空区间。
                if (z["W_infinite_seed_ids"] != empty_ids or
                    z["IS_infinite_seed_ids"] != empty_ids): return "U_AUDIT"
            n_inf = bool(detail["N"]["IS_infinite_seed_count"])
            r_inf = bool(detail["R"]["IS_infinite_seed_count"])
            if kind == "B" and r["any_N_R_width_infinite"] != (n_inf or r_inf):
                return "U_AUDIT"
            if kind == "H3" and (r["N_IS_infinite"] != n_inf or r["R_IS_infinite"] != r_inf):
                return "U_AUDIT"
            # O3/O6 另对明细与逐行来源、摘要和原子表的一致性独立核验。
    if r["units"] != UNITS[kind]: return "U_AUDIT"
    return None

def scientific_tree(kind, r):
    if kind == "A":
        bands = [(83.3,87.3),(87.9,91.9),(2.6,6.6),
                 (16.74,20.46),(20.97,25.63),(3.2,6.2)]
        codes = []
        for j in range(6):  # 长度已在入口精确核查，禁止 zip 截断
            t, (a, b) = r["A_vector"][j], bands[j]
            if t < a: codes.append(f"F_LOW_A_{j+1}")
            elif t > b: codes.append(f"F_HIGH_A_{j+1}")
        return ("失败", codes) if codes else ("成功", ["S_A"])
    if kind == "B":
        c, g, w = r["ci_R_cover"], r["ci_gain"], r["ci_width"]
        if c[1] < 87: return "失败", ["F_LOW_B_C"]
        if c[0] > 93: return "失败", ["F_HIGH_B_C"]
        if g[1] <= 0: return "失败", ["F_NONPOS_B_GAIN"]
        if w is not None and w[1] <= 0:
            return "失败", ["F_NONPOS_B_WIDTH"]
        if r["any_N_R_width_infinite"]:
            return "不确定", ["U_EMPTY_WIDTH"]
        if c[0] >= 87 and c[1] <= 93 and g[0] > 0 and w[0] > 0:
            return "成功", ["S_B"]
        return "不确定", ["U_CI_BOUNDARY"]
    if kind == "H3":
        if r["R_IS_infinite"] and not r["N_IS_infinite"]:
            return "失败", ["F_IS_R_INFINITY"]
        if r["R_IS_infinite"] and r["N_IS_infinite"]:
            return "不确定", ["U_INCOMPARABLE"]
        if r["N_IS_infinite"]:
            return "不确定", ["U_EMPTY_N"]
        d = r["ci_IS_diff"]
        if d[1] < 0: return "成功", ["S_IS_IMPROVE"]
        if d[0] > 0: return "失败", ["F_IS_WORSE"]
        return "不确定", ["U_IS_ZERO"]
    if kind == "C":
        c, g = r["ci_group_cover"], r["ci_group_gain"]
        if c[1] < -2 or g[1] < -2:
            return "失败", ["F_GROUP_DEGRADE"]
        if c[0] >= -2 and g[0] >= -2:
            return "成功", ["S_GROUP_NO_LARGE_DROP"]
        return "不确定", ["U_GROUP_BOUNDARY"]
    return "不确定", ["U_AUDIT"]

def judge(kind, r):
    try:
        code = contract_gate(kind, r)
        if code: return "不确定", [code]
        if not r["object_audit_pass"]: return "不确定", ["U_AUDIT"]
        if not r["locked_environment_match"]:
            return "不确定", ["U_ENV"]
        if kind == "A" and not r["exact_environment"]:
            return "不确定", ["U_ENV"]
        status, codes = scientific_tree(kind, r)
        if kind in ("B", "H3", "C") and not r["exact_environment"]:
            codes = codes + ["E_LOCKED_NONPAPER"]
        return status, codes
    except Exception:
        # 仅封闭契约外输入解析异常，附不可覆盖的异常类型/字段证据；
        # 不重跑、不据此补值，不捕获资源终止来伪造已完成对象。
        return "不确定", ["U_AUDIT"]
