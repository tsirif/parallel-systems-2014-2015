#!/bin/bash
#
#   codingstyle.sh (or short cs)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   A little script to automatically apply my preferred coding style
#   to all .c, .cpp, .h, .java and .py files in and below the current directory
#   Just copy it into a directory that is in your PATH and rename the file to cs.
#
#   Author: Christian Jann <christian 0x2e jann 0x40 ymail 0x2e com>
#   URL: http://www.jann.cc/2013/02/20/code_formatting_using_shell_script.html
#
#   Requirements:
#
#     * astyle (http://astyle.sourceforge.net/astyle.html)
#     * autopep8 (http://pypi.python.org/pypi/autopep8/)
#       (only if it finds Python files)
#

ASTYLE=astyle
ASTYLE_OPTIONS=" --indent-classes --style=kr --indent=spaces=2 --indent-switches --indent-col1-comments --pad-oper --align-pointer=name --remove-brackets --indent-preprocessor --indent-preproc-define --indent-col1-comments --break-blocks --pad-header"
ASTYLE_CLI_OPTIONS=" --suffix=none"


echo $0": Formating source files..."
# Modified kdelibs coding style as defined in
#   http://techbase.kde.org/Policies/Kdelibs_Coding_Style

find -regex ".*\.\(c\|cpp\|h\|java\)" -exec \
    $ASTYLE $ASTYLE_OPTIONS $ASTYLE_CLI_OPTIONS "{}" \;

# Apply coding conventions for Python code
for file in $(find . -name "*.py")
do
  #echo $0": Creating backup: $file.orig"
  #cp -v $file{,.orig}
  cp $file{,.orig}

  #echo $0": Formating file: $file"
  autopep8 -i "$file"

  diff "$file" "$file.orig" >> /dev/null  \
    && echo "Unchanged  $file" || "Formatted  $file"
  rm "$file.orig"
done
