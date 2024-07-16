from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

rectangle = [0, 0, 0, 0]
is_target_lost = True

@app.route('/rectangle', methods=['POST'])
def update_rectangle_position():
    global rectangle
    data = request.get_json()
    target_lost = data.get('target_lost')
    rectangle_lt_x = data.get('rectangle_lt_x')
    rectangle_lt_y = data.get('rectangle_lt_y')
    rectangle_rb_x = data.get('rectangle_rb_x')
    rectangle_rb_y = data.get('rectangle_rb_y')
    if rectangle_lt_x is None or rectangle_lt_y is None or rectangle_rb_x is None or rectangle_rb_y is None or target_lost is None:
        return jsonify({"error": "Invalid data"}), 400
    else:
        is_target_lost = target_lost.lower() == 'true'
        if not is_target_lost:
            rectangle = [float(rectangle_lt_x), 
                         float(rectangle_lt_y), 
                         float(rectangle_rb_x), 
                         float(rectangle_rb_y)]
    
    print("Received data: ", rectangle, is_target_lost)
    return jsonify({"message": "Data received"}), 200

@app.route('/rectangle', methods=['GET'])
def get_rectangle_position():
    global rectangle
    if rectangle is not None and rectangle != [0, 0, 0, 0]:
        print("Sending data: ", rectangle, is_target_lost)
        return jsonify({'rectangle_lt_x': str(rectangle[0]), 
                        'rectangle_lt_y': str(rectangle[1]), 
                        'rectangle_rb_x': str(rectangle[2]), 
                        'rectangle_rb_y': str(rectangle[3]),
                        'target_lost': str(is_target_lost)}), 200
    else:
        return jsonify({"error": "No data available"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
