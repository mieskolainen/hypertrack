# Python Developer Mini Guide
m.mieskolainen@imperial.ac.uk, 2023


## Google Python Style Guide
```
http://google.github.io/styleguide/pyguide.html
```

## Git


### Branching

Show branches
```
git branch
```

Create a new branch
```
git checkout -b <new-branch>
```

Change to another branch
```
git checkout <another-branch>
```

Create a branch from commit hash
```
git branch <branch-name> <hash>
```

### Commits

Add all files
```
git add -A
```

Undo latest local commit (not yet pushed)
```
git reset HEAD~
```

Commit message (add " [no ci]" to the message to skip github actions)
```
git commit -m "commit message"
```

Push to (github)
```
git push origin <branch-name>
```

Pull from (github)
```
git pull origin <branch-name>
```

### Status

Show log
```
git log --pretty=oneline
```

Show status
```
git status
```

### Risky

[Dangerous] hard undo latest commit (removes updates physically)
```
git reset --hard HEAD~
```


## Profiler in Python
```
python -m cProfile -o out.profile "main.py" --config $CONFIG
snakeviz out.profile # visualize
```
