[![Build Status](https://travis-ci.org/bboe/update_checker.png)](https://travis-ci.org/bboe/update_checker)

# update_checker

A python module that will check for package updates.

### Installation

The update_checker module can be installed via:

    pip install update_checker

### Usage

To simply output when there is a newer version of the `update_checker` package,
you can use the following bit of code:

```python
from update_checker import update_check
update_check('update_checker', '0.0.1')
```

If you need more control, such as performing operations conditionally when
there is an update you can use the following approach:

```python
from update_checker import UpdateChecker
checker = UpdateChecker()
result = checker.check('update_checker', '0.0.1')
if result:  # result is None when an update was not found or a failure occured
   # result is a UpdateResult object that contains the following attributes:
   # * available_version
   # * package_name
   # * running_version
   # * release_date (is None if the information isn't available)
   print(result)
   # Conditionally perform other actions
```


