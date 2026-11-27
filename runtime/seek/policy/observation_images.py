import base64
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

from PIL import Image


JPEG_URL = 'data:image/jpeg;base64,'
PROFILE_LIMIT_WH = {'thumbnail_jpeg': (384, 384), 'native_jpeg': (640, 640)}
DETAILS = ('low', 'high', 'original', 'auto')
LAYOUTS = ('separate', 'triptych')
PANEL_ORDER = ('cam_left_wrist', 'cam_head', 'cam_right_wrist')
PANEL_WH = (640, 480)
STRIP_WH = (len(PANEL_ORDER) * PANEL_WH[0], PANEL_WH[1])


@dataclass(frozen=True)
class ImageRule:
    limit_wh: tuple
    exact_wh: tuple | None = None


def image_rule(profile='thumbnail_jpeg', detail='low', image_layout='separate'):
    if profile not in tuple(PROFILE_LIMIT_WH):
        raise ValueError(f'Unknown image profile: {profile}')
    if detail not in DETAILS:
        raise ValueError(f'Unknown image detail: {detail}')
    if image_layout not in LAYOUTS:
        raise ValueError(f'Unknown image layout: {image_layout}')
    if image_layout == 'separate':
        return ImageRule(PROFILE_LIMIT_WH[profile])
    if profile != 'native_jpeg':
        raise ValueError('The triptych layout needs the native_jpeg profile')
    return ImageRule(STRIP_WH, STRIP_WH)


def jpeg_url(picture, **options):
    stream = BytesIO()
    picture.save(stream, format='JPEG', **options)
    return JPEG_URL + base64.b64encode(stream.getvalue()).decode('ascii')


def _bytes(url, strict=True):
    return base64.b64decode(url.split(',', 1)[1], validate=strict)


def check_image(url, rule):
    if not isinstance(url, str) or not url.startswith(JPEG_URL):
        raise ValueError('A camera image must be a JPEG base64 data URL')
    try:
        with Image.open(BytesIO(_bytes(url))) as picture:
            fits = all(size <= limit for size, limit in zip(picture.size, rule.limit_wh))
            if rule.exact_wh is not None:
                fits = fits and tuple(picture.size) == tuple(rule.exact_wh)
            if picture.format != 'JPEG' or picture.mode != 'RGB' or not fits:
                raise ValueError('expected an RGB JPEG within the profile size')
            picture.load()
    except (OSError, ValueError) as error:
        raise ValueError(f'Invalid camera image: {error}') from error


def check_prompt_image(url, profile='thumbnail_jpeg', detail='low', image_layout='separate'):
    check_image(url, image_rule(profile, detail, image_layout))


def _thumbnail(url):
    with Image.open(BytesIO(_bytes(url, strict=False))) as source:
        reduced = source.convert('RGB')
        reduced.thumbnail(PROFILE_LIMIT_WH['thumbnail_jpeg'])
        return jpeg_url(reduced, quality=80)


def prompt_image(archive_url, profile='thumbnail_jpeg'):
    rule = image_rule(profile)
    if profile == 'thumbnail_jpeg':
        return _thumbnail(archive_url)
    check_image(archive_url, rule)
    return archive_url


@lru_cache(maxsize=128)
def _strip(urls):
    panel = ImageRule(PROFILE_LIMIT_WH['native_jpeg'], PANEL_WH)
    canvas = Image.new('RGB', STRIP_WH)
    for index, url in enumerate(urls):
        check_image(url, panel)
        with Image.open(BytesIO(_bytes(url))) as picture:
            canvas.paste(picture, (index * PANEL_WH[0], 0))
    return jpeg_url(canvas, quality=90, subsampling=0)


def _triptych(images, cameras):
    cameras = tuple(cameras)
    if len(cameras) != len(PANEL_ORDER) or set(cameras) != set(PANEL_ORDER):
        raise ValueError('The triptych layout takes each of the three cameras once')
    absent = [camera for camera in PANEL_ORDER if camera not in images]
    if absent:
        raise ValueError(f'No image for {absent}')
    return _strip(tuple(images[camera] for camera in PANEL_ORDER))


def camera_input_parts(step, images, cameras, *, profile='thumbnail_jpeg', detail='low',
                       image_layout='separate'):
    rule = image_rule(profile, detail, image_layout)
    if image_layout == 'triptych':
        url = _triptych(images, cameras)
        check_image(url, rule)
        return [dict(type='input_image', detail=detail, image_url=url)]
    parts = []
    for camera in cameras:
        check_image(images[camera], rule)
        parts.append(dict(type='input_text', text=f'step={step}, camera={camera}'))
        parts.append(dict(type='input_image', detail=detail, image_url=images[camera]))
    return parts
