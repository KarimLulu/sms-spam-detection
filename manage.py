from app import create_app

app = create_app()


@app.cli.command()
def deploy():
    pass

