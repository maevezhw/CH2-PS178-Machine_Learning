from math import sin, radians, sqrt, atan2, cos
import pandas as pd
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from silence_tensorflow import silence_tensorflow
import json


def distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # kilometer

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

def tf_recommend_similar_field(field, current_lat, current_lng):
    # Suppress TensorFlow warnings
    silence_tensorflow()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    field_dataset = pd.read_csv('dataset_cleaned_25Nov (1).csv')

    row_condition = field_dataset['id'] == field
    row_index = field_dataset.index[row_condition][0]
    columns_with_1 = field_dataset.iloc[row_index, 1:] == 1

    columns = ['jenis_sepakbola', 'jenis_badminton', 'jenis_tenis', 'jenis_futsal', 'jenis_voli', 'jenis_basket']
    for i in columns:
        if columns_with_1[i]:
            jenis = i
            break

    pca_field = field_dataset[['lat', 'lng', 'rating', 'price', jenis,
                               'fasilitas_wifi', 'fasilitas_parkir_motor', 'fasilitas_parkir_mobil',
                               'fasilitas_wc', 'fasilitas_kantin', 'fasilitas_mushola']]

    pca_field = pca_field[pca_field[jenis] == 1]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(pca_field), columns=pca_field.columns, index=pca_field.index)

    # Apply TensorFlow-based PCA
    n_components = 2
    
    input_layer = Input(shape=(len(X_scaled.columns),))
    encoded = Dense(n_components, activation='relu')(input_layer)
    decoded = Dense(len(X_scaled.columns), activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose = 0)
    encoded_data = autoencoder.predict(X_scaled)
    X_pca_tensor = tf.convert_to_tensor(encoded_data, dtype=tf.float32)

    # KMeans clustering
    n_clusters = 4
    initial_clusters = tf.random.shuffle(X_pca_tensor)[:n_clusters]
    kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=n_clusters, use_mini_batch=False)

    def input_fn():
        return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(encoded_data, dtype=tf.float32), num_epochs=1)

    previousCenters = initial_clusters
    for _ in range(20):
        kmeans.train(input_fn)
        clusterCenters = kmeans.cluster_centers()
        previousCenters = clusterCenters

    clusterCenters = kmeans.cluster_centers()
    clusterLabels = list(kmeans.predict_cluster_index(input_fn))

    pca_field['cluster'] = clusterLabels
    cluster = pca_field.iloc[row_index]['cluster']

    rec_index = pca_field.index[pca_field['cluster'] == cluster].tolist()
    list_recommend = field_dataset.iloc[rec_index]

    list_recommend['dist'] = list_recommend.apply(lambda x: distance(current_lat, current_lng, x['lat'], x['lng']), axis=1)

    shortest_distance = list_recommend.sort_values(by=['dist'], ascending=True).iloc[0:10]

    output = shortest_distance.to_dict(orient='records')
    json_output = json.dumps(output, indent = 4)

    print(json_output)
    return json.loads(json_output)
