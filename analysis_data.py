"""
Pembangun `analysis_frame` bersama untuk notebook 06-10.

Logika feature engineering identik dengan 05_feature_analysis.ipynb
(join master table + agregasi scan/route/invoice + rasio + interaksi),
tetapi dapat dipakai untuk split "train" maupun "test" dan hasilnya
di-cache ke parquet agar notebook berikutnya tidak perlu membangun ulang.

Pemakaian di notebook:
    from analysis_data import load_analysis_frame
    analysis_frame = load_analysis_frame("train")
    test_analysis_frame = load_analysis_frame("test")
"""

from pathlib import Path

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "Task-1-Datavidia-11.0"
CACHE_DIR = PROJECT_DIR / "cache"

TARGET = "status_penanganan"
ID_COL = "shipment_id"

SHIPMENT_COLUMNS = [
    "shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id",
    "package_id", "promo_id", "weight_kg", "volume_l", "declared_value",
    "distance_km", "ship_month", "ship_dow", "ship_hour", "is_cod", "is_promo",
    "channel", "pickup_type", "goods_category", "dest_region", "insurance_opt",
    "qty_items", "coupon_count", "gift_wrap", "fragility_flag",
    "order_to_ship_hours", "prioritas_pengiriman", "jumlah_catatan",
    "flag_alamat_baru", "berat_volumetrik", "kota_asal", "kota_tujuan",
    "skor_prioritas_kirim", "indeks_rute_sekunder", "faktor_muatan_internal",
    "varian_cuaca_rute", "indeks_waktu_proses", "skor_kualitas_cek",
]

LOOKUPS = [
    # (table, key, features, prefix)
    ("customers", "customer_id", [
        "region", "segment", "tenure_months", "credit_score", "is_business",
        "preferred_payment", "loyalty_tier", "address_quality",
        "complaint_index", "is_verified", "kota_tipe", "metode_notifikasi",
        "tagihan_langganan",
    ], "cust_"),
    ("couriers", "courier_id", [
        "hub_id", "vehicle_type", "experience_years", "contract_type",
        "courier_score", "shift_code", "training_level",
        "jarak_domisili_hub_km", "kapasitas_muatan_kg", "sertifikasi_hazardous",
    ], "cour_"),
    ("hubs", "hub_id", [
        "region", "hub_type", "automation_level", "capacity_index",
        "density_index", "luas_gudang_m2", "jam_buka", "jam_tutup",
    ], "hub_"),
    ("service_packages", "package_id", [
        "speed_class", "handling_class", "insurance_default", "cod_allowed",
        "cost_multiplier", "max_weight_kg", "fragile_score", "estimasi_hari",
        "biaya_dasar",
    ], "pkg_"),
    ("promotions", "promo_id", [
        "nama_promo", "tipe_promo", "diskon_persen", "bulan_mulai",
        "bulan_akhir",
    ], "promo_"),
]

EVENT_TYPES = ["pickup", "sortasi", "transit", "keluar_hub", "upaya_antar", "selesai"]
MODES = ["darat", "laut", "udara", "kereta"]
INVOICE_TYPES = ["pengiriman", "langganan", "klaim"]
PAYMENT_STATUSES = ["lunas", "tertunda", "gagal"]


def read_table(split, table, columns=None):
    return pl.read_csv(
        DATA_DIR / split / f"{table}.csv",
        columns=columns,
        infer_schema_length=20000,
    )


def _scan_agg(split):
    scan = read_table(split, "scan_events", [
        "shipment_id", "hub_id", "event_type", "gap_hours",
        "device_type", "scan_quality", "is_manual",
    ])
    exprs = [
        pl.len().alias("sc_count"),
        pl.col("event_type").n_unique().alias("sc_unique_event"),
        pl.col("hub_id").n_unique().alias("sc_unique_hubs"),
        pl.col("gap_hours").mean().alias("sc_gap_mean"),
        pl.col("gap_hours").median().alias("sc_gap_median"),
        pl.col("gap_hours").max().alias("sc_gap_max"),
        pl.col("gap_hours").sum().alias("sc_gap_sum"),
        pl.col("gap_hours").std().alias("sc_gap_std"),
        pl.col("scan_quality").mean().alias("sc_quality_mean"),
        pl.col("scan_quality").median().alias("sc_quality_median"),
        pl.col("scan_quality").min().alias("sc_quality_min"),
        pl.col("scan_quality").std().alias("sc_quality_std"),
        pl.col("is_manual").cast(pl.Float64, strict=False).sum().alias("sc_manual_count"),
        pl.col("is_manual").cast(pl.Float64, strict=False).mean().alias("sc_manual_rate"),
        pl.col("device_type").n_unique().alias("sc_unique_devices"),
    ]
    for event in EVENT_TYPES:
        exprs.append((pl.col("event_type") == event).sum().alias(f"sc_n_{event}"))
    return scan.group_by("shipment_id").agg(exprs)


def _route_agg(split):
    route = read_table(split, "route_segments", [
        "shipment_id", "urutan", "moda", "jarak_segmen_km",
        "durasi_jam", "segmen_hub_id", "biaya_segmen",
    ]).sort(["shipment_id", "urutan"])
    exprs = [
        pl.len().alias("rt_count"),
        pl.col("jarak_segmen_km").sum().alias("rt_distance_sum"),
        pl.col("jarak_segmen_km").mean().alias("rt_distance_mean"),
        pl.col("jarak_segmen_km").max().alias("rt_distance_max"),
        pl.col("durasi_jam").sum().alias("rt_duration_sum"),
        pl.col("durasi_jam").mean().alias("rt_duration_mean"),
        pl.col("durasi_jam").max().alias("rt_duration_max"),
        pl.col("durasi_jam").null_count().alias("rt_duration_missing"),
        pl.col("biaya_segmen").sum().alias("rt_cost_sum"),
        pl.col("biaya_segmen").mean().alias("rt_cost_mean"),
        pl.col("segmen_hub_id").n_unique().alias("rt_unique_hubs"),
        pl.col("moda").n_unique().alias("rt_unique_modes"),
        pl.col("moda").first().alias("rt_first_mode"),
        pl.col("moda").last().alias("rt_last_mode"),
        (pl.col("jarak_segmen_km") / pl.col("durasi_jam").clip(lower_bound=0.01))
        .mean().alias("rt_speed_mean"),
    ]
    for mode in MODES:
        is_mode = pl.col("moda") == mode
        exprs.extend([
            is_mode.sum().alias(f"rt_n_{mode}"),
            pl.col("durasi_jam").filter(is_mode).sum().alias(f"rt_duration_{mode}"),
            pl.col("jarak_segmen_km").filter(is_mode).sum().alias(f"rt_distance_{mode}"),
        ])
    # maintain_order agar first()/last() mengikuti urutan segmen
    return route.group_by("shipment_id", maintain_order=True).agg(exprs)


def _invoice_agg(split):
    inv = read_table(split, "invoices", [
        "shipment_id", "invoice_type", "amount", "discount", "tax",
        "payment_status", "days_to_pay", "is_penalty",
    ])
    exprs = [
        pl.len().alias("iv_count"),
        pl.col("amount").sum().alias("iv_amount_sum"),
        pl.col("amount").mean().alias("iv_amount_mean"),
        pl.col("amount").max().alias("iv_amount_max"),
        pl.col("discount").sum().alias("iv_discount_sum"),
        pl.col("discount").null_count().alias("iv_discount_missing"),
        pl.col("tax").sum().alias("iv_tax_sum"),
        pl.col("days_to_pay").mean().alias("iv_days_mean"),
        pl.col("days_to_pay").max().alias("iv_days_max"),
        pl.col("is_penalty").cast(pl.Float64, strict=False).sum().alias("iv_penalty_count"),
    ]
    for t in INVOICE_TYPES:
        is_t = pl.col("invoice_type") == t
        exprs.extend([
            is_t.sum().alias(f"iv_n_{t}"),
            pl.col("amount").filter(is_t).sum().alias(f"iv_amount_{t}"),
        ])
    for s in PAYMENT_STATUSES:
        exprs.append((pl.col("payment_status") == s).sum().alias(f"iv_n_{s}"))
    return inv.group_by("shipment_id").agg(exprs)


def build_analysis_frame(split):
    """Bangun analysis_frame untuk split 'train' atau 'test' (logika = notebook 05)."""
    if split not in ("train", "test"):
        raise ValueError("split harus 'train' atau 'test'")

    available = pl.read_csv(DATA_DIR / split / "shipments.csv", n_rows=0).columns
    columns = [c for c in SHIPMENT_COLUMNS if c in available]
    shipments = read_table(split, "shipments", columns)
    n_rows = shipments.height

    frame = shipments
    for table, key, features, prefix in LOOKUPS:
        lookup = read_table(split, table, [key] + features).rename(
            {f: f"{prefix}{f}" for f in features}
        )
        frame = frame.join(lookup, on=key, how="left", maintain_order="left")

    for agg in (_scan_agg(split), _route_agg(split), _invoice_agg(split)):
        frame = frame.join(agg, on="shipment_id", how="left", maintain_order="left")

    frame = frame.with_columns([
        pl.col("promo_nama_promo").fill_null("TANPA_PROMO"),
        pl.col("promo_tipe_promo").fill_null("TANPA_PROMO"),
    ])

    frame = frame.with_columns([
        (pl.col("declared_value").clip(lower_bound=0) + 1).log().alias("log_declared_value"),
        (pl.col("declared_value") / pl.col("weight_kg").clip(lower_bound=0.01)).alias("r_value_per_kg"),
        (pl.col("berat_volumetrik") / pl.col("weight_kg").clip(lower_bound=0.01)).alias("r_volumetric_weight"),
        (pl.col("hub_density_index") / pl.col("hub_capacity_index").clip(lower_bound=0.01)).alias("r_hub_load"),
        (pl.col("rt_duration_sum") / (pl.col("pkg_estimasi_hari") * 24).clip(lower_bound=0.01)).alias("r_duration_estimation"),
        (pl.col("rt_distance_sum") - pl.col("distance_km")).alias("x_route_distance_difference"),
        (pl.col("fragility_flag") * pl.col("pkg_fragile_score")).alias("x_fragility"),
        (pl.col("flag_alamat_baru") * pl.col("jumlah_catatan")).alias("x_address_risk"),
    ])

    # Fitur interaksi (05 cell "interaction_specs")
    frame = frame.with_columns([
        (pl.col("distance_km") * pl.col("pkg_fragile_score")).alias("x_distance_fragility"),
        (pl.col("log_declared_value") * pl.col("insurance_opt").cast(pl.Float64, strict=False)).alias("x_value_insurance"),
        (pl.col("jumlah_catatan") * pl.col("flag_alamat_baru")).alias("x_address_notes"),
        (pl.col("r_hub_load") * pl.col("rt_duration_sum")).alias("x_hub_load_duration"),
        (pl.col("distance_km") * pl.col("rt_count")).alias("x_distance_route_complexity"),
        (pl.col("weight_kg") * pl.col("pkg_fragile_score")).alias("x_weight_fragility"),
    ])

    # Hemat RAM (8 GB): Boolean -> Int8, Float64 -> Float32
    frame = frame.with_columns(
        [pl.col(c).cast(pl.Int8) for c, d in frame.schema.items() if d == pl.Boolean]
        + [pl.col(c).cast(pl.Float32) for c, d in frame.schema.items() if d == pl.Float64]
    )

    assert frame.height == n_rows, "Join menggandakan baris"
    assert frame["shipment_id"].n_unique() == n_rows, "shipment_id tidak unik"
    return frame


def load_analysis_frame(split, rebuild=False):
    """Ambil analysis_frame dari cache parquet; bangun dulu jika belum ada."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"analysis_frame_{split}.parquet"
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    frame = build_analysis_frame(split)
    frame.write_parquet(path)
    return frame


if __name__ == "__main__":
    for s in ("train", "test"):
        f = load_analysis_frame(s, rebuild=True)
        print(s, f.shape, f"{f.estimated_size('mb'):.0f} MB")
