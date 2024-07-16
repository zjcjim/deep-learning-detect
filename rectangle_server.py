from flask import Flask, request, jsonify

app = Flask(__name__)

rectangle = [0, 0, 0, 0]
is_target_lost = True

@app.route('/rectangle', methods=['POST'])
def update_rectangle_position():
    global rectangle
    data = request.get_json()
    is_target_lost = (data.get('target_lost').lower() == 'true')
    if not is_target_lost:
        rectangle[0] = float(data.get('rectangle_lt_x'))
        rectangle[1] = float(data.get('rectangle_lt_y'))
        rectangle[2] = float(data.get('rectangle_rb_x'))
        rectangle[3] = float(data.get('rectangle_rb_y'))
    
    print("Received data: ", rectangle, is_target_lost)

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
