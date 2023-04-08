from flask import Flask, render_template, request
from main import Find_SPI_Values

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    #img, crop, final, center, p_detect, p_spi, possible_count_values, possible_spi_values, _ = \
    #    Find_SPI_Values("main_image/image0.jpg")
    return render_template('home.html', style_url='styles.css')


@app.route('/live')
def about():
    return render_template('live.html', style_url='about.css')


@app.route('/services')
def services():
    return render_template('services.html', style_url='services.css')


@app.route('/gallery')
def gallery():
    return render_template('gallery.html', style_url='gallery.css')


@app.route('/feedback')
def feedback():
    return render_template('feedback.html', style_url='feedback.css')


@app.route('/upload', methods=['POST'])
def upload():
    image = request.files.get('image')
    if image:
        filename = image.filename
        image.save(f'static/images/{filename}')
        IMAGE_PATH = f'static/images/{filename}'
        try:
            image, ci, fi, cc, most_possible_value, most_possible_SPI, possible_count_values, possible_spi_values, total_time = Find_SPI_Values(IMAGE_PATH)
            text_send = "SPI count: " + str(most_possible_SPI)
        except:
            text_send = "Upload better image"

        return text_send
    else:
        return 'No image uploaded'

