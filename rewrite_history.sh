#!/bin/bash
git filter-branch --env-filter '
if [ "$GIT_COMMIT" = "1cf4a805a91a62db8efd944acfb7c53c3ebef6e8" ] || \
   [ "$GIT_COMMIT" = "22e9a660d0fabab1f5bac5bf5c433e6de28df82f" ] || \
   [ "$GIT_COMMIT" = "7a64eac31d49d888f420062c4ff5b6ddf5b1e55f" ]; then
    export GIT_AUTHOR_NAME="falodunos"
    export GIT_AUTHOR_EMAIL="falodunosolomon@gmail.com"
    export GIT_COMMITTER_NAME="falodunos"
    export GIT_COMMITTER_EMAIL="falodunosolomon@gmail.com"
fi
if [ "$GIT_COMMIT" = "5a4ddbb8c3941cd10aac653760d15e366252fed2" ] || \
   [ "$GIT_COMMIT" = "7881cbace3fd5478bfbb9bdb7533ab66faa3fb57" ] || \
   [ "$GIT_COMMIT" = "485b48a8928918389d2e310513d3ebd86bfc3aac" ] || \
   [ "$GIT_COMMIT" = "c9394ace893900252391da139a538d7ea9e975f3" ]; then
    export GIT_AUTHOR_NAME="SilasAmisi"
    export GIT_AUTHOR_EMAIL="swawire@gmail.com"
    export GIT_COMMITTER_NAME="SilasAmisi"
    export GIT_COMMITTER_EMAIL="swawire@gmail.com"
fi
' --tag-name-filter cat -- --branches --tags
