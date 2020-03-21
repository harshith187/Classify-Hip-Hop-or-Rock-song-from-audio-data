from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

tracks = pd.read_csv('fma-rock-vs-hiphop.csv')
echonest_metrics = pd.read_json('echonest-metrics.json')
echo_tracks = pd.merge(echonest_metrics, tracks[["track_id", "genre_top"]])

hop_only = echo_tracks.loc[echo_tracks["genre_top"] == "Hip-Hop"]
rock_only = echo_tracks.loc[echo_tracks["genre_top"] == "Rock"].sample(len(hop_only), random_state=42)
rock_hop_bal = pd.concat([rock_only, hop_only])

lab_enc = LabelEncoder()
rock_hop_bal["genre_top_new"] = lab_enc.fit_transform(rock_hop_bal["genre_top"])
rock_hop_bal["genre_top_new"] = rock_hop_bal["genre_top_new"].astype(float)

X = rock_hop_bal.drop(columns= ["track_id", "genre_top","genre_top_new"])
y = rock_hop_bal["genre_top_new"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_transformed = pca.fit_transform(X_scaled)

X_train,X_test,y_train,y_test = train_test_split(X_transformed, y,random_state = 42)

logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train,y_train)
y_predicted = logreg.predict(X_test)

val_score = cross_val_score(logreg,X_transformed,y,cv = 10)

print(np.mean(val_score))
print(logreg.score(X_test,y_test))