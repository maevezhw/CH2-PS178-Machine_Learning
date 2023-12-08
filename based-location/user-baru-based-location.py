import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from math import cos, asin, sqrt, pi, sin, radians, atan2
from keras.models import load_model
from sklearn.metrics import silhouette_score

def recommend_location(file_path, lng, lat, input_field):
    # Fungsi untuk menghitung jarak antara dua titik berdasarkan koordinat
    def distance(lat1, lon1, lat2, lon2):
        R = 6371.0  # kilometer
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance

    # Fungsi untuk memuat data
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    # Fungsi untuk menemukan jumlah cluster optimal dengan KMeans
    def find_optimal_k(coordinates):
        distortions = []
        K = range(1, 25)
        for k in K:
            kmeansModel = KMeans(n_clusters=k)
            kmeansModel = kmeansModel.fit(coordinates)
            distortions.append(kmeansModel.inertia_)

        sil = []
        kmax = 50

        # dissimilarity would not be defined for a single cluster, thus, the minimum number of clusters should be 2
        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(coordinates)
            labels = kmeans.labels_
            sil.append(silhouette_score(coordinates, labels, metric='euclidean'))

        kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
        kmeans.fit(coordinates)
        y = kmeans.labels_

        return kmeans

    # Fungsi untuk mengelompokkan data ke dalam klaster
    def cluster_data(df, kmeans):
        df['cluster'] = kmeans.predict(df[['lng', 'lat']])
        y_cluster = pd.get_dummies(df['cluster'])
        return y_cluster

    def jenis_lapangan(df, input_field):
        if input_field == 'sepak bola':
            df_sepakbola = df[df['jenis_sepakbola'] == 1]
            return df_sepakbola
        elif input_field == 'voli':
            df_voli = df[df['jenis_voli'] == 1]
            return df_voli
        elif input_field == 'futsal':
            df_futsal = df[df['jenis_futsal'] == 1]
            return df_futsal
        elif input_field == 'badminton':
            df_badminton = df[df['jenis_badminton'] == 1]
            return df_badminton
        elif input_field == 'basket' or input_field == 'tenis':
            df_basket = df[df['jenis_basket'] == 1]
            df_tenis = df[df['jenis_tenis'] == 1]
            df_lainnya = pd.concat([df_basket, df_tenis], axis=0)
            return df_lainnya
        else:
            return "There is nothing"

    # Memuat data
    df = load_data('./Dataset/Data Lapangan.csv')

    # Ambil koordinat
    coordinates = df[['lng', 'lat']]

    # Temukan jumlah cluster optimal dengan KMeans
    kmeans = find_optimal_k(coordinates)

    # Klasterisasi data
    y_cluster = cluster_data(df, kmeans)

    # Bagi data untuk pelatihan dan pengujian model
    X_train, X_test, y_train, y_test = train_test_split(coordinates, y_cluster, test_size=0.2, random_state=42)

    # Bangun model rekomendasi
    model = models.load_model('user-baru_based-on-location.h5')

    # Jenis lapangan
    lapangan_df = jenis_lapangan(df, input_field)

    # Lakukan rekomendasi berdasarkan lokasi
    user_recommendations = recommend_location(lapangan_df, model, lng, lat, input_field)

    json_output = json.dumps(user_recommendations, indent = 4)

    print(json_output)
    return json.loads(json_output)
