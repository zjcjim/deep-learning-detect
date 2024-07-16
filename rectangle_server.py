from flask import Flask, request, jsonify

app = Flask(__name__)

# 用于存储矩形位置的全局变量
rectangle = [0, 0]

@app.route('/rectangle', methods=['POST'])
def update_rectangle_position():
    global rectangle
    data = request.json
    if 'rectangle_lt' in data and 'rectangle_rb' in data:
        rectangle = data
        print("Received data: ", data)
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"error": "Invalid data"}), 400

@app.route('/rectangle', methods=['GET'])
def get_rectangle_position():
    global rectangle
    if rectangle is not None and rectangle != [0, 0]:
        print("Sending data: ", rectangle)
        return jsonify({'rectangle_lt': str(rectangle[0]), 'rectangle_rb': str(rectangle[1])}), 200
    else:
        return jsonify({"error": "No data available"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
