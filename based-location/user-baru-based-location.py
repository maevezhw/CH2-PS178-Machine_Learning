import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import silhouette_score
from math import cos, asin, sqrt, pi, sin, radians, atan2
import json


def homepage_recommend(lng, lat, input_field):
    # Load dataset
    df = pd.read_csv('./Dataset/Data Lapangan.csv')

    # Extract coordinates
    coordinates = df[['lng', 'lat']]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
    df['cluster'] = kmeans.fit_predict(coordinates)

    # Define features and target
    X = coordinates.values
    y = df['cluster'].values

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from keras.models import load_model
    model = load_model('new-user-location.h5')

    def jenis_lapangan(input_field):
        if input_field == 'semua':
            df_semua = df
            return df_semua
        elif input_field == 'sepak bola':
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

    lapangan_df = jenis_lapangan(input_field)

    def distance(lat1, lon1, lat2, lon2):
        R = 6371.0 # kilometer

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    # Extract recommendations based on a given location
    def recommend_location_neural_network(df, model, scaler, lng, lat):
        # Standardize the input coordinates
        input_coordinates = np.array([[lng, lat]])
        input_coordinates = scaler.transform(input_coordinates)

        # Predict the cluster using the neural network model
        predicted_cluster = np.argmax(model.predict(input_coordinates))

        # Filter the dataframe based on the predicted cluster
        cluster_df = df[df['cluster'] == predicted_cluster].copy()

        # Sort the dataframe based on distance (you may use your own distance function here)
        cluster_df['distance'] = cluster_df.apply(lambda x: distance(lat, lng, x['lat'], x['lng']), axis=1)
        sorted_df = cluster_df.sort_values(by=['distance'])

        # Return the top 10 recommendations
        recommendations = sorted_df.iloc[0:10][['name', 'lng', 'lat', 'distance']]
        return recommendations

    recommendations = recommend_location_neural_network(lapangan_df, model, scaler, lng, lat)

    output = recommendations.to_dict(orient='records')
    json_output = json.dumps(output, indent = 4)

    print(json_output)
    return json.loads(json_output)
