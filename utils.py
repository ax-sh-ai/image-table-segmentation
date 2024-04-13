from cv2 import Mat
from pic import Pic


def make_image(src: Mat):
    url = Pic.data_url(src)
    return f'<img src="{url}">'


style = '''<style>
    .overlay-container img, .overlay-container div, .overlay-container section{
        margin:0px !important;
        padding:0px !important;
    }
    img{
        width:100%;
        height:100%;
        object-fit:contain;
    }
    .overlay-container{
        width: 100%;
        background-color: red;
        position:relative
    }
    .overlay{
        position:absolute;
        inset:0px;
        opacity:.7;
    }
</style>'''


def make_overlay(underlay: Mat, overlay: Mat):
    html_string = f"""
{style}
<section class="overlay-container">
    <div class="underlay">{make_image(underlay)}</div>
    <div class="overlay">{make_image(overlay)}</div>
</section>
    """
    return html_string
