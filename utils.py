from cv2 import Mat

from pic import Pic

style = '''<style>.overlay-container {
         position: relative;
         width: 30px;
         height: 30px;
         background-color: red;
      }
    .overlay{
        position: absolute;
        inset:0px;
        opacity: 1;
    }
</style>'''


def make_image(src: Mat):
    url = Pic.data_url(src)
    return f'<img src="{url}">'


style = '''<style>
    img, div, section{
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
        opacity:.8;
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
