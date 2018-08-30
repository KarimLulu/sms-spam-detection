# SMS Spam Detection
The final project for the course ["Data Science. NLP"](https://github.com/vseloved/prj-nlp). The main goal of the project is to build a system for a Ukrainian user.

## Data Collection
The data was collected in multiple ways and labeled manually since there is no open dataset with labeled SMS messages from Ukrainian users. Here are the results:

* 269 responses for [Google Form](https://docs.google.com/forms/d/18Uh1cJIkXQg6_UBJ63u0HGQFy2uKvWTOCZtUYGFZ4U0/) with ~62% spam rate
* ~5800 self-annotated SMS messages from multiple devices with ~18% spam rate

So, here is basic statistics about the final data set:

* ~6100 SMS messages
* ~20% spam rate (imbalancement)
* Two classes (spam and ham)
* Multiple languages - Ukrainian, Russian, translit, English

## Pipeline
