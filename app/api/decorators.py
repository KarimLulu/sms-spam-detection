import logging
import sys


log = logging.getLogger('HTTP Request')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(handler)
log.setLevel(logging.INFO)

def post_response(f):
    def wrapped(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            log.error(str(e))
            return {'status': 'fail', 'message': 'Something went wrong'}, 500
    return wrapped
