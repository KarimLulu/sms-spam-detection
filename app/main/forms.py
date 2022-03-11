from flask_wtf import FlaskForm
from wtforms import TextAreaField, BooleanField, SubmitField


class SmsForm(FlaskForm):
    text = TextAreaField('', validators=[],
                         render_kw={"placeholder": "Enter SMS..."})
    prefill = BooleanField("Prefill with previous sms")
    submit = SubmitField('Submit')
