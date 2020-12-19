from flask_wtf import FlaskForm
from wtforms import StringField, SelectField
from wtforms.validators import DataRequired

class SearchForm(FlaskForm):
    search = StringField('search', validators=[DataRequired()])