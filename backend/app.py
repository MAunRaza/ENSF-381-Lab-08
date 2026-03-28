from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

SEEDED_USERS = {
    "1": {"id": "1", "first_name": "Ava", "user_group": 11},
    "2": {"id": "2", "first_name": "Ben", "user_group": 22},
    "3": {"id": "3", "first_name": "Chloe", "user_group": 33},
    "4": {"id": "4", "first_name": "Diego", "user_group": 44},
    "5": {"id": "5", "first_name": "Ella", "user_group": 55},
}

MODEL_PATH = Path(__file__).resolve().parent / "src" / "random_forest_model.pkl"
PREDICTION_COLUMNS = [
    "city",
    "province",
    "latitude",
    "longitude",
    "lease_term",
    "type",
    "beds",
    "baths",
    "sq_feet",
    "furnishing",
    "smoking",
    "cats",
    "dogs",
]

app = Flask(__name__)
CORS(app)
users = deepcopy(SEEDED_USERS)


@app.route("/users", methods=["GET"])
def get_users():
    return jsonify(list(users.values())), 200


@app.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()

    if not data:
        return jsonify({"message": "Invalid request body."}), 400

    user_id = data.get("id")
    first_name = data.get("first_name")
    user_group = data.get("user_group")

    if user_id is None or first_name is None or user_group is None:
        return jsonify({"message": "Missing id, first_name, or user_group."}), 400

    user_id = str(user_id)

    if user_id in users:
        return jsonify({"message": f"User {user_id} already exists."}), 409

    users[user_id] = {
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
    }

    return jsonify({
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
        "message": f"Created user {user_id}.",
    }), 201


@app.route("/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.get_json()

    if not data:
        return jsonify({"message": "Invalid request body."}), 400

    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404

    first_name = data.get("first_name")
    user_group = data.get("user_group")

    if first_name is None or user_group is None:
        return jsonify({"message": "Missing first_name or user_group."}), 400

    users[user_id] = {
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
    }

    return jsonify({
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
        "message": f"Updated user {user_id}.",
    }), 200


@app.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404

    del users[user_id]
    return jsonify({"message": f"Deleted user {user_id}."}), 200

@app.route("/predict_house_price", methods=["POST"])
def predict_house_price():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"message": "Invalid request body."}), 400

        required_fields = [
            "city",
            "province",
            "latitude",
            "longitude",
            "lease_term",
            "type",
            "beds",
            "baths",
            "sq_feet",
            "furnishing",
            "smoking",
            "pets",
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"message": f"Missing required field: {field}."}), 400

        try:
            latitude = float(data["latitude"])
        except (TypeError, ValueError):
            return jsonify({"message": "latitude must be a number."}), 400

        try:
            longitude = float(data["longitude"])
        except (TypeError, ValueError):
            return jsonify({"message": "longitude must be a number."}), 400

        try:
            beds = float(data["beds"])
        except (TypeError, ValueError):
            return jsonify({"message": "beds must be a number."}), 400

        try:
            baths = float(data["baths"])
        except (TypeError, ValueError):
            return jsonify({"message": "baths must be a number."}), 400

        try:
            sq_feet = float(data["sq_feet"])
        except (TypeError, ValueError):
            return jsonify({"message": "sq_feet must be a number."}), 400

        pets_value = data["pets"]

        if isinstance(pets_value, bool):
            pets = pets_value
        elif isinstance(pets_value, str):
            pets = pets_value.strip().lower() == "true"
        else:
            pets = bool(pets_value)

        cats = pets
        dogs = pets

        model = joblib.load(MODEL_PATH)

        sample_data = [
            str(data["city"]),
            str(data["province"]),
            latitude,
            longitude,
            str(data["lease_term"]),
            str(data["type"]),
            beds,
            baths,
            sq_feet,
            str(data["furnishing"]),
            str(data["smoking"]),
            cats,
            dogs,
        ]

        sample_df = pd.DataFrame([sample_data], columns=PREDICTION_COLUMNS)

        predicted_price = model.predict(sample_df)
        predicted_price = float(predicted_price[0])

        return jsonify({"predicted_price": predicted_price}), 200

    except Exception as error:
        return jsonify({"message": str(error)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5050)