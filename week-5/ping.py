from flask import Flask

app = Flask('ping')

#"what will happen if I send GET request to /ping"
@app.route('/ping', methods = ['GET'])
def ping():
    return "PONG"

def main():
    app.run(debug=True, host='0.0.0.0',port=9696)

#Often used to include code that should only run when the script is executed directly, not when it is imported as a module. 
if __name__=='__main__':
    main()
