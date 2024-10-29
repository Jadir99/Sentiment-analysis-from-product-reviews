# test
import main
from flask import Flask ,render_template,request
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search_url',methods=["POST","GET"])
def search_url():
    url=request.form['url']
    elements,ratings,Product_link=main.scrappingData(url)
    product_name=main.save_df(elements,ratings,url)
    positivs,neutral,negative=main.processing(product_name)
    return render_template('result.html', product_name=product_name,positivs=positivs,neutral=neutral,negative=negative,Product_link=Product_link)


@app.route('/result',methods=["POST","GET"])
def result():
    return render_template('result.html')
if __name__ == '__main__':
    app.run(debug=True,port=4000,use_reloader=False)