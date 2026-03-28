import os
import base64
from IPython.core.display import HTML
from IPython.core.display_functions import display

def show_welcome():
    import base64
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    image_path = os.path.join(filename, 'docs/images/cavsim2d_logo.svg')

    # Convert the image to base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    # HTML and CSS to display image and colourful text on the same line
    message = f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_image}" style="height: 32px;">
        <p style="margin: 0; font-size: 16px; 
                  background: -webkit-linear-gradient(left, #EFC3CA, #5DE2E7, #FE9900, #E7DDFF, #FFDE59);
                  -webkit-background-clip: text; 
                  -webkit-text-fill-color: transparent;">
            <b>CAV-SIM-2D</b> loaded successfully!
        </p>
    </div>
    """

    # Display the HTML
    display(HTML(message))
