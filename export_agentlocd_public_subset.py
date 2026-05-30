from pathlib import Path

from db_config import get_clickhouse_client

# =========================
# 1. 输出文件配置
# =========================
# Save the CSV in the same folder as this Python script.
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "location_info_complete.csv"

# =========================
# 2. 统计总记录数
# =========================
def count_all_records(client):
    query = """
    SELECT count()
    FROM opensource.location_info
    """
    return client.query(query).result_rows[0][0]

# =========================
# 3. 查询与导出
# =========================
def export_location_info(client):
    query = """
    SELECT
        location,
        country,
        administrative_area_level_1,
        administrative_area_level_2,
        locality
    FROM opensource.location_info
    WHERE status = 'normal'
      AND trim(BOTH ' ' FROM ifNull(location, '')) != ''
      AND trim(BOTH ' ' FROM ifNull(country, '')) != ''
      AND trim(BOTH ' ' FROM ifNull(administrative_area_level_1, '')) != ''
      AND trim(BOTH ' ' FROM ifNull(administrative_area_level_2, '')) != ''
      AND trim(BOTH ' ' FROM ifNull(locality, '')) != ''
    """
    df = client.query_df(query)
    return df

# =========================
# 4. 主函数
# =========================
def main():
    client = get_clickhouse_client()
    print("Connected to ClickHouse")

    total_records = count_all_records(client)
    print(f"Total records in opensource.location_info: {total_records}")

    df = export_location_info(client)
    print(f"Total complete records with status='normal' and non-empty fields: {len(df)}")

    # 保存到 CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved filtered records to {OUTPUT_CSV}")

    # 打印前 5 条预览
    print(df.head())

if __name__ == "__main__":
    main()