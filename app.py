# import the necessary packages

from flask import Flask
from flask import request
from flask import send_file
import io

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import convertToJPG

from deoldify.visualize import *
from pathlib import Path
import traceback


torch.backends.cudnn.benchmark = True


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)


# define a predict function as an endpoint
@app.route("/process", methods=["POST"])
def process_image():

    input_path = generate_random_filename(upload_directory,"jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    try:
        url = request.json["source_url"]
        render_factor = int(request.json["render_factor"])

        download(url, input_path)

        try:
            image_colorizer.plot_transformed_image(path=input_path, figsize=(20, 20),
                                                   render_factor=render_factor, watermarked=False)
        except:
            convertToJPG(input_path)
            image_colorizer.plot_transformed_image(path=input_path, figsize=(20, 20),
                                                   render_factor=render_factor, watermarked=False)

        return_data = io.BytesIO()
        with open(output_path, 'rb') as fo:
            return_data.write(fo.read())

        return_data.seek(0)

        callback = send_file(return_data, mimetype='image/jpeg')
        
        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        pass
        clean_all([
            input_path,
            output_path
            ])


if __name__ == '__main__':

    upload_directory = 'data/upload/'
    create_directory(upload_directory)

    results_img_directory = 'data/result_images/'
    create_directory(results_img_directory)

    model_directory = 'data/models/'
    create_directory(model_directory)

    artistic_model_url = 'https://www.dropbox.com/s/zkehq1uwahhbc2o/ColorizeStable_gen.pth?dl=0'
    get_model_bin(artistic_model_url, os.path.join(model_directory, 'ColorizeStable_gen.pth'))

    image_colorizer = get_stable_image_colorizer(root_folder=Path('./data'))
    image_colorizer.results_dir = Path(results_img_directory)
    
    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=False)
