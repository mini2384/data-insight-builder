import re
import math
import numpy as np
import pandas as pd

# 집계 행(대륙, 소득 그룹, 연합체 등) 탐지 키워드
AGG_KEYWORDS = [
    "world", "oecd members", "euro area", "european union",
    "high income", "low income", "middle income", "income",
    "area", "total", "dividend", "small states",
    "ida & ibrd", "ibrd only", "ida only", "ida total", "not classified"
]


def sanitize_for_json(obj):
    """NaN, Inf, -Inf 값을 JSON 호환 가능한 None 또는 안전한 값으로 재귀 변환"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    return obj


def filter_aggregates_if_needed(df, exclude_aggregates=True, label_col=None):
    """
    집계 행(대륙, 소득군 등)을 제외하는 공통 헬퍼 함수.
    반환: (필터링된_df, 제외된_행_수, 제외_이유)
    """
    if not exclude_aggregates:
        return df, 0, "집계 행을 포함하여 전체 데이터를 사용했습니다."

    if label_col is None:
        # 문자열 열 중 첫 번째를 레이블 컬럼으로 자동 선정
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_col = text_cols[0] if text_cols else df.columns[0]

    mask = df[label_col].astype(str).str.lower().apply(
        lambda val: any(kw in val for kw in AGG_KEYWORDS)
    )
    excluded_cnt = int(mask.sum())
    filtered_df = df[~mask].copy()

    reason = f"대륙·소득군 등 집계용 그룹 행 {excluded_cnt}개를 제외하여 순수 국가/항목만 남겼습니다." if excluded_cnt > 0 else "집계 행이 감지되지 않아 제외된 행이 없습니다."
    return filtered_df, excluded_cnt, reason


# =====================================================================
# 도구 1: reshape_wide_to_long (연도 열을 행으로 접기 / Unpivot)
# =====================================================================
def reshape_wide_to_long(df, id_vars=None, value_vars=None, var_name="Year", value_name="Value", exclude_aggregates=True):
    """
    가로로 펼쳐진 연도 열들을 세로(Long format)로 변환(Unpivot).
    """
    filtered_df, excluded_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates)

    # 4자리 연도 열 자동 탐색
    all_year_cols = [c for c in filtered_df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())]
    if value_vars is None:
        value_vars = all_year_cols
    else:
        value_vars = [c for c in value_vars if c in filtered_df.columns]

    if id_vars is None:
        id_vars = [c for c in filtered_df.columns if c not in all_year_cols and not str(c).startswith("Unnamed:")]

    if not value_vars:
        raise ValueError("접을 수 있는 연도(숫자) 열이 데이터에 없습니다.")

    long_df = pd.melt(
        filtered_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name
    )

    # 결측치 수 집계 및 미리보기 생성
    total_cells = len(long_df)
    valid_cells = int(long_df[value_name].notna().sum())
    missing_cells = total_cells - valid_cells

    preview_rows = long_df.dropna(subset=[value_name]).head(20).to_dict(orient="records")

    result = {
        "original_rows": len(filtered_df),
        "total_melted_rows": total_cells,
        "valid_value_rows": valid_cells,
        "missing_value_rows": missing_cells,
        "id_variables": id_vars,
        "value_variables_count": len(value_vars),
        "preview": sanitize_for_json(preview_rows)
    }

    return {
        "tool": "reshape_wide_to_long",
        "result": result,
        "reference_period": f"{value_vars[0]}~{value_vars[-1]}" if value_vars else "전체",
        "used_rows": len(filtered_df),
        "excluded_rows": excluded_cnt,
        "exclusion_reason": excl_reason,
        "used_columns": id_vars + [var_name, value_name],
        "unit": "행 수 (변환된 레코드)",
        "caution": "연도별 결측치(데이터가 없던 해)는 변환 후 빈 값으로 남으므로 분석 시 필터링이 필요합니다."
    }


# =====================================================================
# 도구 2: quality_report (데이터 품질 종합 진단)
# =====================================================================
def quality_report(df, exclude_aggregates=False):
    """
    결측치, 중복, 데이터 타입, 비정상 열 이름 등을 정밀하게 계산한 데이터 품질 보고서.
    """
    filtered_df, excluded_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates)

    total_rows = len(filtered_df)
    total_cols = len(filtered_df.columns)

    # 열별 결측치 통계
    missing_stats = []
    high_missing_cols = []
    for col in filtered_df.columns:
        cnt = int(filtered_df[col].isna().sum())
        rate = round(cnt / total_rows * 100, 2) if total_rows > 0 else 0
        stat = {"column": str(col), "missing_count": cnt, "missing_rate": rate, "dtype": str(filtered_df[col].dtype)}
        missing_stats.append(stat)
        if rate >= 50.0:
            high_missing_cols.append(str(col))

    # 중복 행
    duplicate_count = int(filtered_df.duplicated().sum())

    # 비정상 열 이름 (Unnamed, 빈 문자열)
    weird_cols = [str(c) for c in filtered_df.columns if "Unnamed:" in str(c) or str(c).strip() == ""]

    # 품질 평점 (0~100점 간이 계산)
    overall_null_cells = int(filtered_df.isna().sum().sum())
    total_cells = total_rows * total_cols if total_rows * total_cols > 0 else 1
    cell_fill_rate = round((1 - overall_null_cells / total_cells) * 100, 1)

    result = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "overall_fill_rate": cell_fill_rate,
        "duplicate_rows": duplicate_count,
        "unnamed_or_empty_columns": weird_cols,
        "columns_with_over_50pct_missing": high_missing_cols,
        "column_missing_details": sanitize_for_json(missing_stats)
    }

    return {
        "tool": "quality_report",
        "result": result,
        "reference_period": "전체 시점",
        "used_rows": total_rows,
        "excluded_rows": excluded_cnt,
        "exclusion_reason": excl_reason,
        "used_columns": [str(c) for c in filtered_df.columns],
        "unit": "건수 및 비율(%)",
        "caution": f"결측치가 50% 이상인 열이 {len(high_missing_cols)}개 존재합니다. 해당 열을 분석에 사용할 때는 주의가 필요합니다."
    }


# =====================================================================
# 도구 3: year_coverage (연도별 데이터 충실도 및 권장 기준연도)
# =====================================================================
def year_coverage(df, exclude_aggregates=True):
    """
    각 연도별로 유효한 데이터가 몇 개 채워져 있는지 계산하고 차트 데이터와 권장 기준연도를 반환.
    """
    filtered_df, excluded_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates)

    year_cols = [c for c in filtered_df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())]
    if not year_cols:
        raise ValueError("데이터에 연도(19xx, 20xx) 열이 존재하지 않습니다.")

    coverage_data = []
    labels = []
    fill_counts = []
    fill_rates = []

    total_items = len(filtered_df)
    for y in sorted(year_cols):
        cnt = int(filtered_df[y].notna().sum())
        rate = round(cnt / total_items * 100, 2) if total_items > 0 else 0
        labels.append(str(y))
        fill_counts.append(cnt)
        fill_rates.append(rate)
        coverage_data.append({
            "year": str(y),
            "filled_count": cnt,
            "total_items": total_items,
            "fill_rate": rate
        })

    # 권장 기준연도 계산 (최근 연도 중 충실도 70% 이상인 가장 최신 연도)
    recommended_year = None
    for item in sorted(coverage_data, key=lambda x: int(x["year"]), reverse=True):
        if item["fill_rate"] >= 70.0:
            recommended_year = item["year"]
            break

    if not recommended_year and coverage_data:
        recommended_year = max(coverage_data[-5:], key=lambda x: x["filled_count"])["year"]

    # 차트용 데이터 (Chart.js 규격)
    chart = {
        "type": "bar",
        "labels": labels,
        "datasets": [
            {
                "label": "데이터 충실도 (%)",
                "data": fill_rates,
                "backgroundColor": ["rgba(54, 162, 235, 0.6)" if y != recommended_year else "rgba(75, 192, 192, 0.9)" for y in labels]
            }
        ]
    }

    result = {
        "recommended_year": recommended_year,
        "total_analyzed_items": total_items,
        "coverage_table": sanitize_for_json(coverage_data),
        "chart": chart
    }

    return {
        "tool": "year_coverage",
        "result": result,
        "reference_period": f"{labels[0]}~{labels[-1]} (권장: {recommended_year})",
        "used_rows": total_items,
        "excluded_rows": excluded_cnt,
        "exclusion_reason": excl_reason,
        "used_columns": year_cols,
        "unit": "% (충실도)",
        "caution": f"권장 기준연도는 {recommended_year}년입니다. 가장 마지막 연도는 결측치가 많을 수 있으니 주의하세요."
    }


# =====================================================================
# 도구 4: detect_aggregates (집계 행 후보 탐지)
# =====================================================================
def detect_aggregates(df, label_column=None):
    """
    World, High income, OECD members 등 국가가 아닌 대륙/지역/소득군 집계 행을 탐지하여 목록과 비율 반환.
    """
    if label_column is None:
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_column = text_cols[0] if text_cols else df.columns[0]

    aggregates = []
    pure_items = []

    for idx, row in df.iterrows():
        val = str(row[label_column])
        val_lower = val.lower()
        matched_kw = None
        for kw in AGG_KEYWORDS:
            if kw in val_lower:
                matched_kw = kw
                break

        if matched_kw:
            aggregates.append({
                "row_index": int(idx),
                "name": val,
                "matched_keyword": matched_kw
            })
        else:
            pure_items.append(val)

    total_rows = len(df)
    agg_count = len(aggregates)
    pure_count = len(pure_items)

    chart = {
        "type": "pie",
        "labels": ["순수 개별 국가/항목", "집계 그룹 후보"],
        "datasets": [{
            "data": [pure_count, agg_count],
            "backgroundColor": ["rgba(54, 162, 235, 0.7)", "rgba(255, 99, 132, 0.7)"]
        }]
    }

    result = {
        "label_column": str(label_column),
        "total_rows": total_rows,
        "pure_items_count": pure_count,
        "aggregates_count": agg_count,
        "aggregate_ratio": round(agg_count / total_rows * 100, 2) if total_rows > 0 else 0,
        "detected_aggregates": sanitize_for_json(aggregates),
        "chart": chart
    }

    return {
        "tool": "detect_aggregates",
        "result": result,
        "reference_period": "전체 시점",
        "used_rows": total_rows,
        "excluded_rows": 0,
        "exclusion_reason": "모든 행을 대상으로 집계 그룹 여부를 전수 조사했습니다.",
        "used_columns": [str(label_column)],
        "unit": "행 수 (개)",
        "caution": "집계 행이 포함된 상태로 전체 평균을 구하면 중복 계산(Double Counting)으로 인해 통계가 왜곡됩니다."
    }


# =====================================================================
# 도구 5: describe_numeric (기초 통계량 계산)
# =====================================================================
def describe_numeric(df, column=None, reference_period=None, exclude_aggregates=True):
    """
    개수, 평균, 중앙값, 최소, 최대, 표준편차, 4분위수를 계산.
    """
    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates)

    if reference_period and str(reference_period) in filtered_df.columns:
        target_col = str(reference_period)
    elif column and str(column) in filtered_df.columns:
        target_col = str(column)
    else:
        year_cols = [c for c in filtered_df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())]
        target_col = year_cols[-1] if year_cols else None
        if not target_col:
            num_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])]
            target_col = num_cols[0] if num_cols else None

    if not target_col:
        raise ValueError("통계를 계산할 숫자형 열을 찾을 수 없습니다.")

    series = pd.to_numeric(filtered_df[target_col], errors="coerce")
    valid_series = series.dropna()

    missing_in_col = len(series) - len(valid_series)
    total_excluded = excluded_agg_cnt + missing_in_col
    combined_reason = f"{excl_reason} 또한 '{target_col}' 열에 값이 없는 결측치 {missing_in_col}개를 계산에서 제외했습니다."

    if len(valid_series) == 0:
        raise ValueError(f"'{target_col}' 열에 유효한 숫자 데이터가 없습니다.")

    count_val = int(len(valid_series))
    mean_val = round(float(valid_series.mean()), 2)
    std_val = round(float(valid_series.std()), 2) if count_val > 1 else 0.0
    median_val = round(float(valid_series.median()), 2)
    min_val = round(float(valid_series.min()), 2)
    max_val = round(float(valid_series.max()), 2)
    q25_val = round(float(valid_series.quantile(0.25)), 2)
    q75_val = round(float(valid_series.quantile(0.75)), 2)

    chart = {
        "type": "bar",
        "labels": ["최소값", "25% 분위", "중앙값", "평균값", "75% 분위", "최대값"],
        "datasets": [{
            "label": f"{target_col} 요약 통계",
            "data": [min_val, q25_val, median_val, mean_val, q75_val, max_val],
            "backgroundColor": [
                "rgba(255, 99, 132, 0.6)",
                "rgba(255, 159, 64, 0.6)",
                "rgba(75, 192, 192, 0.8)",
                "rgba(54, 162, 235, 0.8)",
                "rgba(153, 102, 255, 0.6)",
                "rgba(75, 192, 192, 0.6)"
            ]
        }]
    }

    result = {
        "column": str(target_col),
        "count": count_val,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "q25": q25_val,
        "q75": q75_val,
        "chart": chart
    }

    return {
        "tool": "describe_numeric",
        "result": result,
        "reference_period": str(target_col),
        "used_rows": count_val,
        "excluded_rows": total_excluded,
        "exclusion_reason": combined_reason,
        "used_columns": [str(target_col)],
        "unit": "수치 (요약 통계량)",
        "caution": "평균값은 극단적인 이상치(극소/극대값)에 영향을 많이 받으므로, 중앙값과 함께 비교하여 해석해야 합니다."
    }


# =====================================================================
# 도구 6: trend_line (선택 항목들의 연도별 시계열 추이 선 차트)
# =====================================================================
def trend_line(df, items=None, label_column=None, year_columns=None, exclude_aggregates=True):
    """
    선택된 국가/항목들의 연도별 변화 추이를 꺾은선 차트로 시각화.
    """
    if label_column is None:
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_column = text_cols[0] if text_cols else df.columns[0]

    all_years = sorted([c for c in df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())])
    if year_columns is None:
        year_columns = [y for y in all_years if int(y) >= 1990]
        if not year_columns:
            year_columns = all_years

    available_names = df[label_column].astype(str).tolist()
    if not items:
        candidates = ["Korea, Rep.", "United States", "Japan", "Germany", "United Kingdom", "World"]
        items = [c for c in candidates if c in available_names]
        if not items:
            items = available_names[:4]

    matched_rows = []
    for item in items:
        row_match = df[df[label_column].astype(str) == str(item)]
        if not row_match.empty:
            matched_rows.append(row_match.iloc[0])

    if not matched_rows:
        raise ValueError(f"선택한 항목({items})을 데이터에서 찾을 수 없습니다.")

    color_palette = [
        "rgba(54, 162, 235, 1)",
        "rgba(255, 99, 132, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(255, 159, 64, 1)",
        "rgba(153, 102, 255, 1)",
        "rgba(201, 203, 207, 1)"
    ]

    datasets = []
    item_summaries = {}
    for idx, row in enumerate(matched_rows):
        name = str(row[label_column])
        series_vals = [sanitize_for_json(float(row[y])) if pd.notna(row[y]) and str(row[y]).strip() != "" else None for y in year_columns]
        valid_vals = [v for v in series_vals if v is not None]

        first_val = valid_vals[0] if valid_vals else None
        last_val = valid_vals[-1] if valid_vals else None
        growth = round(last_val - first_val, 2) if first_val is not None and last_val is not None else None

        item_summaries[name] = {
            "start_value": first_val,
            "latest_value": last_val,
            "total_change": growth
        }

        color = color_palette[idx % len(color_palette)]
        datasets.append({
            "label": name,
            "data": series_vals,
            "borderColor": color,
            "backgroundColor": color.replace("1)", "0.1)"),
            "fill": False,
            "tension": 0.2
        })

    chart = {
        "type": "line",
        "labels": year_columns,
        "datasets": datasets
    }

    result = {
        "selected_items": items,
        "year_range": f"{year_columns[0]}~{year_columns[-1]}",
        "item_summaries": item_summaries,
        "chart": chart
    }

    return {
        "tool": "trend_line",
        "result": result,
        "reference_period": f"{year_columns[0]}~{year_columns[-1]}",
        "used_rows": len(matched_rows),
        "excluded_rows": len(df) - len(matched_rows),
        "exclusion_reason": f"선택된 {len(matched_rows)}개 항목을 제외한 나머지 행은 추이 비교에서 제외되었습니다.",
        "used_columns": [str(label_column)] + year_columns,
        "unit": "수치 (시계열 변화량)",
        "caution": "연도별 결측치가 존재하는 항목은 선 그래프의 해당 구간이 끊어지거나 비어있을 수 있습니다."
    }


# =====================================================================
# 도구 7: top_bottom_n (기준연도 기준 상위·하위 N 순위 수평 막대)
# =====================================================================
def top_bottom_n(df, reference_period=None, n=10, label_column=None, mode="both", exclude_aggregates=True):
    """
    기준 시점에 수치가 존재하는 항목 중 상위/하위 N개 순위를 산출.
    기준 시점(reference_period)은 필수 파라미터.
    """
    if not reference_period:
        raise ValueError("기준 시점(reference_period)은 필수 입력 항목입니다. (예: '2024')")

    ref_col = str(reference_period)
    if ref_col not in df.columns:
        raise ValueError(f"데이터에 지정된 기준 시점 열('{ref_col}')이 존재하지 않습니다.")

    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates, label_column)

    if label_column is None:
        text_cols = [c for c in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_column = text_cols[0] if text_cols else filtered_df.columns[0]

    numeric_series = pd.to_numeric(filtered_df[ref_col], errors="coerce")
    valid_mask = numeric_series.notna()
    missing_cnt = int((~valid_mask).sum())

    working_df = filtered_df[valid_mask].copy()
    working_df["_val"] = numeric_series[valid_mask]

    total_excluded = excluded_agg_cnt + missing_cnt
    reasons = []
    if excluded_agg_cnt > 0:
        reasons.append(f"집계 행 {excluded_agg_cnt}개 제외")
    if missing_cnt > 0:
        reasons.append(f"기준연도({ref_col}) 결측치 행 {missing_cnt}개 제외")
    full_excl_reason = f"{', '.join(reasons)}되어 총 {len(working_df)}개 행을 분석에 사용했습니다." if reasons else "제외된 행 없이 분석을 수행했습니다."

    if working_df.empty:
        raise ValueError(f"기준 시점 '{ref_col}'에 유효한 데이터가 단 하나도 없습니다.")

    working_df.sort_values(by="_val", ascending=False, inplace=True)

    top_df = working_df.head(n)
    bottom_df = working_df.tail(n).sort_values(by="_val", ascending=True)

    top_items = [{"rank": idx + 1, "name": str(row[label_column]), "value": round(float(row["_val"]), 2)} for idx, (_, row) in enumerate(top_df.iterrows())]
    bottom_items = [{"rank": len(working_df) - idx, "name": str(row[label_column]), "value": round(float(row["_val"]), 2)} for idx, (_, row) in enumerate(bottom_df.iterrows())]

    if mode == "top":
        chart_labels = [item["name"] for item in top_items]
        chart_data = [item["value"] for item in top_items]
        bg_colors = ["rgba(75, 192, 192, 0.7)"] * len(chart_data)
    elif mode == "bottom":
        chart_labels = [item["name"] for item in bottom_items]
        chart_data = [item["value"] for item in bottom_items]
        bg_colors = ["rgba(255, 99, 132, 0.7)"] * len(chart_data)
    else:
        chart_labels = [item["name"] for item in top_items] + ["---"] + [item["name"] for item in bottom_items]
        chart_data = [item["value"] for item in top_items] + [None] + [item["value"] for item in bottom_items]
        bg_colors = ["rgba(75, 192, 192, 0.7)"] * len(top_items) + ["rgba(0,0,0,0)"] + ["rgba(255, 99, 132, 0.7)"] * len(bottom_items)

    chart = {
        "type": "bar",
        "indexAxis": "y",
        "labels": chart_labels,
        "datasets": [{
            "label": f"{ref_col}년 수치",
            "data": chart_data,
            "backgroundColor": bg_colors
        }]
    }

    result = {
        "reference_period": ref_col,
        "n": n,
        "mode": mode,
        "total_valid_countries": len(working_df),
        "top_items": top_items,
        "bottom_items": bottom_items,
        "chart": chart
    }

    return {
        "tool": "top_bottom_n",
        "result": result,
        "reference_period": ref_col,
        "used_rows": len(working_df),
        "excluded_rows": total_excluded,
        "exclusion_reason": full_excl_reason,
        "used_columns": [str(label_column), ref_col],
        "unit": "수치 (순위별 값)",
        "caution": f"오직 {ref_col}년 시점에 데이터가 존재하는 국가들만 동일 시점으로 순위를 매겼습니다. (다른 연도와 혼합하지 않음)"
    }


# =====================================================================
# 도구 8: group_summary (그룹별 통계 및 비교 막대 차트)
# =====================================================================
def group_summary(df, group_column=None, value_column=None, reference_period=None, agg_func="mean", exclude_aggregates=True):
    """
    지역(Region), 소득수준(IncomeGroup) 등 그룹별로 평균·합계·개수를 집계.
    기준 시점(reference_period)은 필수 파라미터.
    """
    import glob

    if not reference_period:
        raise ValueError("기준 시점(reference_period)은 필수 입력 항목입니다. (예: '2024')")

    val_col = str(reference_period) if value_column is None else str(value_column)
    if val_col not in df.columns:
        raise ValueError(f"데이터에 기준 시점 또는 값 열('{val_col}')이 존재하지 않습니다.")

    working_df = df.copy()
    if group_column in ["Region", "IncomeGroup", None] and "Country Code" in working_df.columns:
        if group_column not in working_df.columns:
            meta_files = glob.glob("data/**/Metadata_Country*.csv", recursive=True)
            if meta_files:
                try:
                    meta_df = pd.read_csv(meta_files[0], encoding="utf-8-sig")
                    join_cols = [c for c in ["Country Code", "Region", "IncomeGroup"] if c in meta_df.columns]
                    working_df = pd.merge(working_df, meta_df[join_cols], on="Country Code", how="left")
                    if group_column is None:
                        group_column = "Region"
                except Exception:
                    pass

    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(working_df, exclude_aggregates)

    if group_column is None or group_column not in filtered_df.columns:
        cat_candidates = [c for c in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        if cat_candidates:
            suitable_groups = [c for c in cat_candidates if 2 <= filtered_df[c].nunique() <= 20]
            group_column = suitable_groups[0] if suitable_groups else cat_candidates[0]
        else:
            raise ValueError("그룹화할 범주형 열(group_column)을 찾을 수 없습니다.")

    numeric_vals = pd.to_numeric(filtered_df[val_col], errors="coerce")
    valid_mask = numeric_vals.notna() & filtered_df[group_column].notna() & (filtered_df[group_column].astype(str).str.strip() != "")

    missing_cnt = int((~valid_mask).sum())
    analyzed_df = filtered_df[valid_mask].copy()
    analyzed_df["_numeric_val"] = numeric_vals[valid_mask]

    total_excluded = excluded_agg_cnt + missing_cnt
    full_excl_reason = f"{excl_reason} 또한 그룹 정보가 없거나 '{val_col}'에 값이 없는 행 {missing_cnt}개를 제외했습니다."

    if analyzed_df.empty:
        raise ValueError(f"그룹별 집계를 수행할 유효 데이터가 없습니다. ('{group_column}', '{val_col}')")

    grouped = analyzed_df.groupby(group_column)["_numeric_val"].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        sum="sum"
    ).reset_index()

    grouped.sort_values(by="mean", ascending=False, inplace=True)

    group_stats = []
    labels = []
    chart_vals = []

    for _, row in grouped.iterrows():
        g_name = str(row[group_column])
        cnt = int(row["count"])
        m_val = round(float(row["mean"]), 2)
        med_val = round(float(row["median"]), 2)
        s_val = round(float(row["sum"]), 2)
        min_v = round(float(row["min"]), 2)
        max_v = round(float(row["max"]), 2)

        labels.append(g_name)
        val_to_plot = m_val if agg_func == "mean" else (s_val if agg_func == "sum" else med_val)
        chart_vals.append(val_to_plot)

        group_stats.append({
            "group": g_name,
            "count": cnt,
            "mean": m_val,
            "median": med_val,
            "sum": s_val,
            "min": min_v,
            "max": max_v
        })

    chart = {
        "type": "bar",
        "labels": labels,
        "datasets": [{
            "label": f"그룹별 {agg_func.upper()} ({val_col})",
            "data": chart_vals,
            "backgroundColor": "rgba(54, 162, 235, 0.7)"
        }]
    }

    result = {
        "group_column": str(group_column),
        "value_column": val_col,
        "agg_func": agg_func,
        "reference_period": val_col,
        "total_analyzed_rows": len(analyzed_df),
        "group_count": len(grouped),
        "group_details": group_stats,
        "chart": chart
    }

    return {
        "tool": "group_summary",
        "result": result,
        "reference_period": val_col,
        "used_rows": len(analyzed_df),
        "excluded_rows": total_excluded,
        "exclusion_reason": full_excl_reason,
        "used_columns": [str(group_column), val_col],
        "unit": "수치 (그룹별 요약치)",
        "caution": "그룹 내 속한 국가 수가 적은 그룹(예: 3개국 이하)은 극단적인 값 1개로 인해 그룹 평균이 크게 왜곡될 수 있습니다."
    }


# =====================================================================
# 도구 9: change_rate (두 시점 사이 변화량·변화율)
# =====================================================================
def change_rate(df, start_period=None, end_period=None, reference_period=None, label_column=None, n=10, mode="top_growth", exclude_aggregates=True):
    """
    두 시점(예: 2014년 vs 2024년) 사이의 절대 변화량(%p)과 상대 변화율(%)을 계산.
    기준 시점(reference_period)은 필수 파라미터.
    """
    if not reference_period and not end_period:
        raise ValueError("기준 시점(reference_period)은 필수 입력 항목입니다. (예: '2024' 또는 '2014-2024')")

    if reference_period and "-" in str(reference_period):
        parts = str(reference_period).split("-")
        start_period, end_period = parts[0].strip(), parts[1].strip()
    elif reference_period and not end_period:
        end_period = str(reference_period)

    ref_col = str(end_period)
    if start_period is None:
        all_years = sorted([c for c in df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())])
        if ref_col in all_years:
            ref_idx = all_years.index(ref_col)
            start_period = all_years[max(0, ref_idx - 10)]
        else:
            start_period = all_years[0] if all_years else None

    if not start_period or str(start_period) not in df.columns or str(end_period) not in df.columns:
        raise ValueError(f"시작 시점('{start_period}') 또는 종료 시점('{end_period}') 열을 데이터에서 찾을 수 없습니다.")

    start_col = str(start_period)
    end_col = str(end_period)

    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates, label_column)

    if label_column is None:
        text_cols = [c for c in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_column = text_cols[0] if text_cols else filtered_df.columns[0]

    s_num = pd.to_numeric(filtered_df[start_col], errors="coerce")
    e_num = pd.to_numeric(filtered_df[end_col], errors="coerce")

    valid_mask = s_num.notna() & e_num.notna()
    missing_cnt = int((~valid_mask).sum())

    working_df = filtered_df[valid_mask].copy()
    working_df["_start"] = s_num[valid_mask]
    working_df["_end"] = e_num[valid_mask]
    working_df["_diff"] = working_df["_end"] - working_df["_start"]
    working_df["_rate"] = np.where(working_df["_start"] > 0, ((working_df["_end"] - working_df["_start"]) / working_df["_start"]) * 100, 0.0)

    total_excluded = excluded_agg_cnt + missing_cnt
    full_excl_reason = f"{excl_reason} 또한 두 시점({start_col}, {end_col}) 중 하나라도 값이 비어 있는 {missing_cnt}개 행을 제외했습니다."

    if working_df.empty:
        raise ValueError(f"두 시점('{start_col}' 및 '{end_col}')에 모두 값이 존재하는 행이 없습니다.")

    working_df.sort_values(by="_diff", ascending=False, inplace=True)

    top_growth = working_df.head(n)

    top_list = [{
        "rank": idx + 1,
        "name": str(row[label_column]),
        "start_val": round(float(row["_start"]), 2),
        "end_val": round(float(row["_end"]), 2),
        "abs_change": round(float(row["_diff"]), 2),
        "rel_change_pct": round(float(row["_rate"]), 2)
    } for idx, (_, row) in enumerate(top_growth.iterrows())]

    chart_labels = [item["name"] for item in top_list]
    chart_data = [item["abs_change"] for item in top_list]

    chart = {
        "type": "bar",
        "labels": chart_labels,
        "datasets": [{
            "label": f"{start_col}년 대비 {end_col}년 증가량 (%p)",
            "data": chart_data,
            "backgroundColor": "rgba(75, 192, 192, 0.7)"
        }]
    }

    result = {
        "start_period": start_col,
        "end_period": end_col,
        "total_analyzed_items": len(working_df),
        "average_change": round(float(working_df["_diff"].mean()), 2),
        "median_change": round(float(working_df["_diff"].median()), 2),
        "top_growth_items": top_list,
        "chart": chart
    }

    return {
        "tool": "change_rate",
        "result": result,
        "reference_period": f"{start_col}~{end_col}",
        "used_rows": len(working_df),
        "excluded_rows": total_excluded,
        "exclusion_reason": full_excl_reason,
        "used_columns": [str(label_column), start_col, end_col],
        "unit": "%p (포인트 변화량) 및 % (변화율)",
        "caution": "초기 시점(시작연도)의 수치가 매우 낮은 국가는 작은 증가로도 변화율(%)이 수천 %로 폭증해 착시를 일으킬 수 있으므로 절대 변화량(%p)을 함께 확인해야 합니다."
    }


# =====================================================================
# 도구 10: distribution (구간별 빈도 분포 / 히스토그램)
# =====================================================================
def distribution(df, reference_period=None, bins=10, exclude_aggregates=True):
    """
    기준 시점의 수치를 10개 구간(예: 0~10%, 10~20%...)으로 나누어 빈도와 분포 형태를 분석.
    기준 시점(reference_period)은 필수 파라미터.
    """
    if not reference_period:
        raise ValueError("기준 시점(reference_period)은 필수 입력 항목입니다. (예: '2024')")

    ref_col = str(reference_period)
    if ref_col not in df.columns:
        raise ValueError(f"데이터에 기준 시점 열('{ref_col}')이 존재하지 않습니다.")

    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates)

    num_series = pd.to_numeric(filtered_df[ref_col], errors="coerce")
    valid_series = num_series.dropna()

    missing_cnt = len(num_series) - len(valid_series)
    total_excluded = excluded_agg_cnt + missing_cnt
    full_excl_reason = f"{excl_reason} 또한 '{ref_col}' 열에 값이 없는 {missing_cnt}개 행을 제외했습니다."

    if valid_series.empty:
        raise ValueError(f"기준 시점 '{ref_col}'에 유효한 숫자 데이터가 없습니다.")

    min_v = float(valid_series.min())
    max_v = float(valid_series.max())
    range_min = 0.0 if min_v >= 0 and max_v <= 100 else math.floor(min_v)
    range_max = 100.0 if max_v <= 100 else math.ceil(max_v)

    counts, bin_edges = np.histogram(valid_series, bins=bins, range=(range_min, range_max))

    bin_labels = []
    bin_details = []
    total_count = len(valid_series)

    for i in range(len(counts)):
        b_start = round(float(bin_edges[i]), 1)
        b_end = round(float(bin_edges[i + 1]), 1)
        cnt = int(counts[i])
        pct = round(cnt / total_count * 100, 1) if total_count > 0 else 0
        label_str = f"{b_start}~{b_end}%"
        bin_labels.append(label_str)
        bin_details.append({
            "bin_range": label_str,
            "min": b_start,
            "max": b_end,
            "count": cnt,
            "percentage": pct
        })

    mode_bin = max(bin_details, key=lambda x: x["count"])

    chart = {
        "type": "bar",
        "labels": bin_labels,
        "datasets": [{
            "label": f"{ref_col}년 국가 수 분포",
            "data": [b["count"] for b in bin_details],
            "backgroundColor": ["rgba(54, 162, 235, 0.6)" if b != mode_bin else "rgba(255, 99, 132, 0.8)" for b in bin_details]
        }]
    }

    result = {
        "reference_period": ref_col,
        "total_analyzed_items": total_count,
        "bin_count": bins,
        "mode_range": mode_bin["bin_range"],
        "mode_count": mode_bin["count"],
        "distribution_table": bin_details,
        "chart": chart
    }

    return {
        "tool": "distribution",
        "result": result,
        "reference_period": ref_col,
        "used_rows": total_count,
        "excluded_rows": total_excluded,
        "exclusion_reason": full_excl_reason,
        "used_columns": [ref_col],
        "unit": "국가/항목 수 (빈도)",
        "caution": "특정 구간(예: 90~100%)에 극단적으로 국가가 몰려 있는 편향된 분포(Skewed)일 수 있으므로 정규분포 가정을 주의해야 합니다."
    }


# =====================================================================
# 도구 11: correlation_scatter (두 숫자 열 상관계수 및 산점도)
# =====================================================================
def correlation_scatter(df, x_column=None, y_column=None, reference_period=None, label_column=None, exclude_aggregates=True):
    """
    두 시점 또는 두 지표 열 간의 피어슨 상관계수(r) 계산 및 산점도(Scatter plot) 데이터 생성.
    기준 시점(reference_period)은 필수 파라미터.
    """
    if not reference_period and not y_column:
        raise ValueError("기준 시점(reference_period)은 필수 입력 항목입니다. (예: '2024')")

    if reference_period and not y_column:
        y_column = str(reference_period)

    ref_col = str(y_column)
    if ref_col not in df.columns:
        raise ValueError(f"데이터에 지정된 기준 시점 열('{ref_col}')이 존재하지 않습니다.")

    all_years = sorted([c for c in df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())])
    if x_column is None:
        if ref_col in all_years:
            ref_idx = all_years.index(ref_col)
            x_column = all_years[max(0, ref_idx - 10)]
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != ref_col]
            x_column = num_cols[0] if num_cols else None

    if not x_column or str(x_column) not in df.columns:
        raise ValueError("비교할 X축 숫자 열을 찾을 수 없습니다.")

    x_col = str(x_column)
    y_col = ref_col

    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates, label_column)

    if label_column is None:
        text_cols = [c for c in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_column = text_cols[0] if text_cols else filtered_df.columns[0]

    x_num = pd.to_numeric(filtered_df[x_col], errors="coerce")
    y_num = pd.to_numeric(filtered_df[y_col], errors="coerce")

    valid_mask = x_num.notna() & y_num.notna()
    missing_cnt = int((~valid_mask).sum())

    working_df = filtered_df[valid_mask].copy()
    working_df["_x"] = x_num[valid_mask]
    working_df["_y"] = y_num[valid_mask]

    total_excluded = excluded_agg_cnt + missing_cnt
    full_excl_reason = f"{excl_reason} 또한 두 열('{x_col}', '{y_col}') 중 결측치가 있는 {missing_cnt}개 행을 제외했습니다."

    if len(working_df) < 2:
        raise ValueError("상관계수를 계산하기 위한 유효 행 수가 부족합니다(최소 2개 이상 필요).")

    corr_r = round(float(working_df["_x"].corr(working_df["_y"])), 4)

    abs_r = abs(corr_r)
    if abs_r >= 0.7:
        strength = "매우 강한 상관관계"
    elif abs_r >= 0.4:
        strength = "뚜렷한 상관관계"
    elif abs_r >= 0.2:
        strength = "약한 상관관계"
    else:
        strength = "거의 상관관계 없음"

    scatter_points = []
    for _, row in working_df.iterrows():
        scatter_points.append({
            "x": round(float(row["_x"]), 2),
            "y": round(float(row["_y"]), 2),
            "name": str(row[label_column])
        })

    chart = {
        "type": "scatter",
        "datasets": [{
            "label": f"{x_col} vs {y_col} 산점도",
            "data": scatter_points,
            "backgroundColor": "rgba(54, 162, 235, 0.6)"
        }]
    }

    result = {
        "x_column": x_col,
        "y_column": y_col,
        "correlation_coefficient": corr_r,
        "correlation_strength": strength,
        "total_analyzed_items": len(working_df),
        "scatter_points_sample": scatter_points[:30],
        "chart": chart
    }

    return {
        "tool": "correlation_scatter",
        "result": result,
        "reference_period": f"{x_col} & {y_col}",
        "used_rows": len(working_df),
        "excluded_rows": total_excluded,
        "exclusion_reason": full_excl_reason,
        "used_columns": [str(label_column), x_col, y_col],
        "unit": "상관계수 (피어슨 r, -1 ~ +1)",
        "caution": "[주의] 상관관계는 인과관계가 아닙니다! 두 변수가 함께 증가한다고 해서 한쪽이 다른 쪽의 원인인 것은 아닙니다."
    }


# =====================================================================
# 도구 12: compare_one_vs_all (특정 항목 vs 전체 평균·중앙값 비교 강조)
# =====================================================================
def compare_one_vs_all(df, target_item=None, reference_period=None, label_column=None, exclude_aggregates=True):
    """
    특정 국가/항목(예: 대한민국)의 수치를 전체 평균, 중앙값, 최상위/최하위 값과 비교하여 강조.
    기준 시점(reference_period)은 필수 파라미터.
    """
    if not reference_period:
        raise ValueError("기준 시점(reference_period)은 필수 입력 항목입니다. (예: '2024')")

    ref_col = str(reference_period)
    if ref_col not in df.columns:
        raise ValueError(f"데이터에 기준 시점 열('{ref_col}')이 존재하지 않습니다.")

    filtered_df, excluded_agg_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates, label_column)

    if label_column is None:
        text_cols = [c for c in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[c]) and not re.match(r"^(19|20)\d{2}$", str(c).strip())]
        label_column = text_cols[0] if text_cols else filtered_df.columns[0]

    all_names = filtered_df[label_column].astype(str).tolist()
    if not target_item:
        if "Korea, Rep." in all_names:
            target_item = "Korea, Rep."
        else:
            target_item = all_names[0] if all_names else None

    if not target_item or target_item not in all_names:
        raise ValueError(f"비교 대상 항목('{target_item}')을 데이터에서 찾을 수 없습니다.")

    num_series = pd.to_numeric(filtered_df[ref_col], errors="coerce")
    valid_mask = num_series.notna()
    missing_cnt = int((~valid_mask).sum())

    working_df = filtered_df[valid_mask].copy()
    working_df["_val"] = num_series[valid_mask]

    target_match = working_df[working_df[label_column].astype(str) == str(target_item)]
    if target_match.empty:
        raise ValueError(f"선택한 대상('{target_item}')은 기준 시점('{ref_col}')에 수치가 비어 있어 비교할 수 없습니다.")

    target_value = round(float(target_match.iloc[0]["_val"]), 2)

    total_excluded = excluded_agg_cnt + missing_cnt
    full_excl_reason = f"{excl_reason} 또한 '{ref_col}' 열에 값이 없는 {missing_cnt}개 행을 제외했습니다."

    all_vals = working_df["_val"]
    mean_val = round(float(all_vals.mean()), 2)
    median_val = round(float(all_vals.median()), 2)
    min_val = round(float(all_vals.min()), 2)
    max_val = round(float(all_vals.max()), 2)

    working_df.sort_values(by="_val", ascending=False, inplace=True)
    working_df.reset_index(drop=True, inplace=True)
    rank_idx = working_df[working_df[label_column].astype(str) == str(target_item)].index[0]
    rank_val = int(rank_idx + 1)
    total_valid = len(working_df)
    percentile_val = round((1 - (rank_val - 1) / total_valid) * 100, 1)

    chart_labels = ["최저치", "전체 평균", "전체 중앙값", f"★ {target_item}", "최고치"]
    chart_data = [min_val, mean_val, median_val, target_value, max_val]
    chart_colors = [
        "rgba(255, 99, 132, 0.6)",
        "rgba(54, 162, 235, 0.6)",
        "rgba(153, 102, 255, 0.6)",
        "rgba(255, 206, 86, 0.9)",
        "rgba(75, 192, 192, 0.6)"
    ]

    chart = {
        "type": "bar",
        "labels": chart_labels,
        "datasets": [{
            "label": f"{ref_col}년 비교 지표",
            "data": chart_data,
            "backgroundColor": chart_colors
        }]
    }

    result = {
        "target_item": str(target_item),
        "target_value": target_value,
        "reference_period": ref_col,
        "rank": rank_val,
        "total_items": total_valid,
        "percentile": percentile_val,
        "overall_mean": mean_val,
        "overall_median": median_val,
        "diff_from_mean": round(target_value - mean_val, 2),
        "diff_from_median": round(target_value - median_val, 2),
        "min_value": min_val,
        "max_value": max_val,
        "chart": chart
    }

    return {
        "tool": "compare_one_vs_all",
        "result": result,
        "reference_period": ref_col,
        "used_rows": total_valid,
        "excluded_rows": total_excluded,
        "exclusion_reason": full_excl_reason,
        "used_columns": [str(label_column), ref_col],
        "unit": "수치 (상대 비교값)",
        "caution": f"비교 대상({target_item})의 위치를 전체 국가 분포와 비교했습니다. 평균보다 높은 것과 중앙값보다 높은 것의 의미 차이를 인지해야 합니다."
    }


# =====================================================================
# 전체 12개 도구 레지스트리 (Gemini 제안 및 API 실행용 카탈로그)
# =====================================================================
TOOL_REGISTRY = {
    "reshape_wide_to_long": reshape_wide_to_long,
    "quality_report": quality_report,
    "year_coverage": year_coverage,
    "detect_aggregates": detect_aggregates,
    "describe_numeric": describe_numeric,
    "trend_line": trend_line,
    "top_bottom_n": top_bottom_n,
    "group_summary": group_summary,
    "change_rate": change_rate,
    "distribution": distribution,
    "correlation_scatter": correlation_scatter,
    "compare_one_vs_all": compare_one_vs_all
}

TOOL_CATALOG = [
    {
        "id": "reshape_wide_to_long",
        "name": "연도 열을 행으로 접기",
        "description": "가로로 펼쳐진 연도 열들을 [국가, 연도, 수치] 형태의 세로(Long) 데이터로 재구조화합니다.",
        "requires_reference_period": False,
        "chart_type": "table"
    },
    {
        "id": "quality_report",
        "name": "데이터 품질 종합 리포트",
        "description": "결측치 비율, 중복 행, 이상한 열 이름 등을 정밀 진단합니다.",
        "requires_reference_period": False,
        "chart_type": "table"
    },
    {
        "id": "year_coverage",
        "name": "연도별 데이터 충실도 분석",
        "description": "연도별 유효 데이터 개수와 충실도(%)를 계산하고 권장 기준연도를 제안합니다.",
        "requires_reference_period": False,
        "chart_type": "bar"
    },
    {
        "id": "detect_aggregates",
        "name": "집계 행 후보 탐지",
        "description": "전 세계(World), 고소득국 등 개별 국가가 아닌 대륙/지역 묶음 행을 식별합니다.",
        "requires_reference_period": False,
        "chart_type": "pie"
    },
    {
        "id": "describe_numeric",
        "name": "기초 통계량 계산",
        "description": "개수, 평균, 중앙값, 최소, 최대, 표준편차, 4분위수를 산출합니다.",
        "requires_reference_period": False,
        "chart_type": "bar"
    },
    {
        "id": "trend_line",
        "name": "시계열 추이 비교 (선 차트)",
        "description": "선택한 주요 국가들의 연도별 변화 흐름을 꺾은선 그래프로 비교합니다.",
        "requires_reference_period": False,
        "chart_type": "line"
    },
    {
        "id": "top_bottom_n",
        "name": "기준연도 상위·하위 N 순위",
        "description": "지정한 기준연도(동일 시점)에 데이터가 있는 국가 중 상위 및 하위 국가 순위를 매깁니다.",
        "requires_reference_period": True,
        "chart_type": "horizontal_bar"
    },
    {
        "id": "group_summary",
        "name": "그룹별 평균·합계 비교",
        "description": "대륙(Region) 또는 소득수준 등 그룹별 평균 및 통계를 비교합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    },
    {
        "id": "change_rate",
        "name": "두 시점 간 변화량·성장률",
        "description": "과거 시점과 현재 시점 사이의 절대 증가량(%p)과 상대 성장률(%)을 비교합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    },
    {
        "id": "distribution",
        "name": "구간별 빈도 분포 (히스토그램)",
        "description": "기준연도 수치의 구간별(예: 0~10%, 10~20%...) 국가 수 분포를 확인합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    },
    {
        "id": "correlation_scatter",
        "name": "상관계수 및 산점도 분석",
        "description": "과거와 현재 또는 두 지표 간의 피어슨 상관계수를 계산하고 산점도를 그립니다.",
        "requires_reference_period": True,
        "chart_type": "scatter"
    },
    {
        "id": "compare_one_vs_all",
        "name": "특정 항목 vs 전체 비교",
        "description": "특정 국가(예: 한국)의 수치를 전체 평균, 중앙값, 최고치와 한눈에 비교합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    }
]
