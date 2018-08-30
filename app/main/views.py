import warnings
warnings.filterwarnings("ignore")

from flask import render_template, make_response, flash

from . import main
from .forms import SmsForm
from src.config import model_id, THRESHOLD, SPAM_LABEL, HAM_LABEL, ROUND
from src.helpers import load_model, prepare_sms

model = load_model(name=model_id)

@main.route('/', methods=['GET', 'POST'])
def detector():
    form = SmsForm()
    headers = {'Content-Type': 'text/html; charset=UTF-8',
               'Cache-Control': 'no-cache, no-store, must-revalidate',
               'Pragma': 'no-cache',
               'Expires': '0',
               'Cache-Control': 'public, max-age=0'}

    if form.validate_on_submit():
        text = form.text.data
        if not form.prefill.data:
            form.text.data = None
        input_data = prepare_sms(text)
        ham_proba, spam_proba = model.predict_proba(input_data)[0]

        if ham_proba >= THRESHOLD:
            label = category = HAM_LABEL
            confidence = ham_proba
        else:
            label = category = SPAM_LABEL
            confidence = spam_proba

        confidence = round(confidence * 100, ROUND)
        flash(f'Label: {label}', category=category)
        flash(f'Confidence: {confidence} %', category=category)
    return make_response(render_template('main.html', form=form), 200, headers)