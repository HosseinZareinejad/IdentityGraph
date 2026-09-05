import pandas as pd
import os

def convert_parquet_to_csv(parquet_path, csv_path=None):
    """
    تبدیل فایل Parquet به CSV با پشتیبانی کامل از کاراکترهای فارسی
    """
    if not os.path.exists(parquet_path):
        print(f"خطا: فایل '{parquet_path}' پیدا نشد.")
        return

    # اگر مسیر خروجی مشخص نشده باشد، هم‌نام با فایل اصلی ذخیره می‌شود
    if csv_path is None:
        csv_path = os.path.splitext(parquet_path)[0] + '.csv'

    print(f"در حال خواندن فایل {parquet_path}...")
    df = pd.read_parquet(parquet_path, engine='pyarrow')

    print(f"در حال تبدیل به {csv_path}...")
    # استفاده از utf-8-sig جهت خوانش درست متون فارسی در نرم‌افزارهایی مانند Excel
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"تبدیل با موفقیت انجام شد! ({len(df)} سطر ذخیره شد)")

if __name__ == "__main__":
    # نام فایل پارکت ورودی
    input_file = "identity_dataset.parquet"
    
    convert_parquet_to_csv(input_file)