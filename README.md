# Joker-Lottery-Models
Try to analyze the joker lottery data statistically and develop ML models for prediction. I am doing it
just for fun (If it was working correctly then I would not do coding anymore:)) and to learn new things about Monte Carlo, Markov Chains, ARIMA model, Random Forest Classifier, and LSTM
models mainly using Pandas dataframes.

## How to Use
The code run multiple deep and statistic analysis and return the results. You can use the code by running the following command:
```shell
joker_lottery_models -vv --year 2025 --day 4 --week 8
```
where `--year` is the year of the lottery, `--day` is the day of the lottery, and `--week` is the week of the lottery. The code will return the results of the analysis.

## How to Develop
Do the following only once after creating your project:
- Init the git repo with `git init`.
- Add files with `git add .`.
- Then `git commit -m 'initialize the project'`.
- Add remote url with `git remote add origin REPO_URL`.
- Then `git branch -M master`.
- `git push origin main`.
Then create a branch with `git checkout -b BRANCH_NAME` for further developments.
- Install poetry if you do not have it in your system from [here](https://python-poetry.org/docs/#installing-with-pipx).
- Create a virtual env preferably with virtualenv wrapper and `mkvirtualenv -p $(which python3.10) ENVNAME`.
- Then `git add poetry.lock`.
- Then `pre-commit install`.
- For applying changes use `pre-commit run --all-files`.

## Docker Container
To run the docker with ssh, do the following first and then based on your need select ,test, development, or production containers:
```shell
export DOCKER_BUILDKIT=1
export DOCKER_SSHAGENT="-v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK -e SSH_AUTH_SOCK"
```

### Test Container
This container is used for testing purposes while it runs the test:
```shell
docker build --progress plain --ssh default --target test -t joker_lottery:test .
docker run -it --rm -v "$(pwd):/app" $(echo $DOCKER_SSHAGENT) joker_lottery:test
```

### Development Container
This container can be used for development purposes:
```shell
docker build --progress plain --ssh default --target development -t joker_lottery:development .
docker run -it --rm -v "$(pwd):/app" -v /tmp:/tmp $(echo $DOCKER_SSHAGENT) joker_lottery:development
```

### Production Container
This container can be used for production purposes:
```shell
docker build --progress plain --ssh default --target production -t joker_lottery:production .
docker run -it --rm -v "$(pwd):/app" -v /tmp:/tmp $(echo $DOCKER_SSHAGENT) joker_lottery:production joker_lottery_models -vv --year 2025 --day 4 --week 8
```
