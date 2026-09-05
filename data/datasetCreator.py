import pandas as pd
import uuid
import random
from faker import Faker
import pyarrow as pa
import pyarrow.parquet as pq
import os

print("Initializing Faker libraries...")
fake_fa = Faker('fa_IR')
fake_en = Faker('en_US')

golden_records = []
num_golden = 5000

print(f"Generating {num_golden} golden records...")
for _ in range(num_golden):
    entity_id = str(uuid.uuid4())
    record_id = str(uuid.uuid4())
    first_name = fake_fa.first_name()
    last_name = fake_fa.last_name()
    full_name = f"{first_name} {last_name}"
    username = fake_en.user_name()
    email = f"{username}@{fake_en.free_email_domain()}"
    bio = fake_fa.text(max_nb_chars=120)
    city = fake_fa.city()
    birth_year = random.randint(1340, 1400)
    phone = fake_fa.phone_number()
    is_noisy = False
    
    golden_records.append({
        'record_id': record_id,
        'entity_id': entity_id,
        'first_name_ref': first_name, # Temporary for advanced noise generation
        'last_name_ref': last_name,   # Temporary for advanced noise generation
        'full_name': full_name,
        'username': username,
        'email': email,
        'bio': bio,
        'city': city,
        'birth_year': birth_year,
        'phone_number': phone,
        'is_noisy': is_noisy
    })

num_to_duplicate = int(num_golden * 0.3)
records_to_duplicate = random.sample(golden_records, num_to_duplicate)
print(f"Applying noise to {num_to_duplicate} selected records to create complex duplicates...")

def apply_noise(record):
    noisy = record.copy()
    noisy['record_id'] = str(uuid.uuid4())
    noisy['is_noisy'] = True
    
    # 1. Missing data (Null injection)
    if random.random() < 0.2:
        noisy['city'] = None
    if random.random() < 0.2:
        noisy['bio'] = None
    if random.random() < 0.15:
        noisy['phone_number'] = None
    if random.random() < 0.1:
        noisy['email'] = None

    # 2. Username format change
    if noisy['username'] and random.random() < 0.5:
        if '_' in noisy['username']:
            noisy['username'] = noisy['username'].replace('_', '')
        else:
            noisy['username'] = noisy['username'] + str(random.randint(10, 999))
            
    # 3. Email typo injection
    if noisy['email'] and random.random() < 0.3:
        parts = noisy['email'].split('@')
        if len(parts) == 2:
            if '.' in parts[0] and random.random() < 0.5:
                noisy['email'] = parts[0].replace('.', '') + '@' + parts[1]
            elif parts[1] == 'gmail.com' and random.random() < 0.5:
                noisy['email'] = parts[0] + '@googlemail.com'
            elif random.random() < 0.5:
                noisy['email'] = parts[0] + '@' + parts[1].replace('.com', '.ir')

    # 4. Complex Name Noise (Typos, Swaps, Titles)
    if noisy['full_name'] and random.random() < 0.7:
        noise_type = random.choices(
            ['replace_y', 'replace_k', 'delete_char', 'swap_first_last', 'add_title', 'phonetic'], 
            weights=[15, 15, 20, 20, 20, 10], k=1
        )[0]
        
        name = noisy['full_name']
        if noise_type == 'replace_y' and 'ی' in name:
            noisy['full_name'] = name.replace('ی', 'ي', 1)
        elif noise_type == 'replace_k' and 'ک' in name:
            noisy['full_name'] = name.replace('ک', 'ك', 1)
        elif noise_type == 'delete_char' and len(name) > 4:
            idx = random.randint(1, len(name)-2)
            if name[idx] != ' ': # don't delete space
                noisy['full_name'] = name[:idx] + name[idx+1:]
        elif noise_type == 'swap_first_last':
            # Swapping Last Name and First Name (Common in Iranian datasets)
            noisy['full_name'] = f"{noisy['last_name_ref']} {noisy['first_name_ref']}"
        elif noise_type == 'add_title':
            titles = ['دکتر', 'مهندس', 'آقای', 'خانم', 'سید', 'سیده']
            noisy['full_name'] = f"{random.choice(titles)} {name}"
        elif noise_type == 'phonetic':
            # Simple phonetic mistakes
            if 'ز' in name: noisy['full_name'] = name.replace('ز', 'ذ', 1)
            elif 'س' in name: noisy['full_name'] = name.replace('س', 'ص', 1)
            elif 'ت' in name: noisy['full_name'] = name.replace('ت', 'ط', 1)

    # 5. Bio spelling noise
    if noisy['bio'] and random.random() < 0.4:
        if 'ی' in noisy['bio']:
            noisy['bio'] = noisy['bio'].replace('ی', 'ي', random.randint(1, 3))
            
    # 6. Phone number format noise
    if noisy['phone_number'] and random.random() < 0.6:
        phone = noisy['phone_number']
        if phone.startswith('+98'):
            noisy['phone_number'] = '0' + phone[3:]
        elif phone.startswith('0'):
            noisy['phone_number'] = '+98' + phone[1:]
        elif random.random() < 0.2:
            # Drop zero completely
            if phone.startswith('0'):
                noisy['phone_number'] = phone[1:]
            
    # 7. Birth year noise (Off-by-one or typo)
    if noisy['birth_year'] and random.random() < 0.15:
        if random.random() < 0.5:
            noisy['birth_year'] += random.choice([-1, 1]) # Close match mistake
        else:
            noisy['birth_year'] += random.choice([-10, 10]) # Decade typo

    return noisy

all_records = []
# Remove temporary fields before saving golden records
for record in golden_records:
    rec_copy = dict(record)
    del rec_copy['first_name_ref']
    del rec_copy['last_name_ref']
    all_records.append(rec_copy)

# Generate and add noisy records
for record in records_to_duplicate:
    num_copies = random.randint(1, 3)
    for _ in range(num_copies):
        noisy_copy = apply_noise(record)
        del noisy_copy['first_name_ref']
        del noisy_copy['last_name_ref']
        all_records.append(noisy_copy)

print(f"Total records generated (Golden + Noisy): {len(all_records)}")
df = pd.DataFrame(all_records)
# Shuffle dataframe to simulate real-world scatter
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = r'e:\project\IdentityGraph\identity_dataset.parquet'
print(f"Saving dataset to {output_path} with Snappy compression...")
table = pa.Table.from_pandas(df)
pq.write_table(table, output_path, compression='snappy')
print("✅ Dataset generation completed successfully!")
