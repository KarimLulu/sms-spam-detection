from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SubmitField
from wtforms import validators

class SmsForm(FlaskForm):
    text = TextAreaField('', validators=[], 
                         render_kw={"placeholder": "Enter SMS..."})
    prefill = BooleanField("Prefill with previous sms")
    submit = SubmitField('Submit')