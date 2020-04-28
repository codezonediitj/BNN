BNN
===

[![Build Status](https://travis-ci.com/codezonediitj/BNN.svg?branch=master)](https://travis-ci.com/codezonediitj/BNN) [![Join the chat at https://gitter.im/codezoned2017/Lobby](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/codezoned2017/Lobby) ![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

About
--------

A simple, light weight deep learning framework with a specialisation aimed for Binarized Neural Networks. 
This project aims to overcome some problems with existing deep learning frameworks without compromising performance, like most of the time they are quite heavy, come with a lot of dependencies, require innumerable extra packages for accelerated computing. This project will just depend on CUDA C for NVIDIA GPUs. In future, support may be added for Intel GPUs.

Technologies
------------

We are using the following technologies in our project,

1. C++
2. Python
3. CUDA C
4. Google Test
5. Boost.Python

How to contribute?
------------------

Follow the steps given below,

1. Fork, https://github.com/codezonediitj/BNN
2. Execute, `git clone https://github.com/codezonediitj/BNN/`
3. Change your working directory to `../BNN`.
4. Execute, `git remote add origin_user https://github.com/<your-github-username>/BNN/`
5. Execute, `git checkout -b <your-new-branch-for-working>`.
6. Make changes to the code.
7. Add your name and email to the AUTHORS, if you wish to.
8. Execute, `git add .`.
9. Execute, `git commit -m "your-commit-message"`.
10. Execute, `git push origin_user <your-current-branch>`.
11. Make a PR.

That's it, 10 easy steps for your first contribution. For future contributions just follow steps 5 to 10. Make sure that before starting work, always checkout to master and pull the recent changes using the remote `origin` and then start following steps 5 to 10.

See you soon with your first PR.

Guidelines
----------

We recommend you to introduce yourself on our [gitter channel](https://gitter.im/codezoned2017/Lobby). You can include the literature you have studied relevant to Binarized Neural Networks, some projects, prior experience with the technologies mentioned above, in your introduction.

Please follow the rules and guidelines given below,

1. For Python we follow the [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).
2. For C++ we follow our own coding style mentioned which is yet to be finalised.
3. For C++ documentation we follow, Doxygen style guide. Refer to various modules in the existing `master` branch for the pattern.
4. Follow the Pull Request policy given [here](https://github.com/codezonediitj/BNN/wiki/Pull-Request-Policy). All changes are made through Pull Requests, no direct commits to the master branch.

Keep contributing!!
