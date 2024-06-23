from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

data = pd.read_csv('data.csv')

label_encoder = LabelEncoder()
data['Klasifikasi Kemiskinan'] = label_encoder.fit_transform(data['Klasifikasi Kemiskinan'])

# Memisahkan fitur dan label
X = data[['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)', 'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)', 'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)', 'Indeks Pembangunan Manusia', 'Umur Harapan Hidup (Tahun)', 'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak', 'Persentase rumah tangga yang memiliki akses terhadap air minum layak', 'Tingkat Pengangguran Terbuka', 'Tingkat Partisipasi Angkatan Kerja', 'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)']]
y = data['Klasifikasi Kemiskinan']

# Membagi data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = [
        float(data['persentase_penduduk_miskin']),
        float(data['rata_rata_lama_sekolah']),
        float(data['pengeluaran_per_kapita']),
        float(data['indeks_pembangunan_manusia']),
        float(data['umur_harapan_hidup']),
        float(data['persentase_sanitasi']),
        float(data['persentase_air_minum']),
        float(data['tingkat_pengangguran']),
        float(data['tingkat_partisipasi_angkatan_kerja']),
        float(data['pdrb'])
    ]
    
    prediction = model.predict([input_data])[0]
    label = 'Tidak Miskin' if prediction == 0 else 'Miskin'
    
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
