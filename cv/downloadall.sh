
# I'm a jerk.
export USERAGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11"

# don't delete this line. we need it
# so bash doesn't replace our tabs with spaces.
export IFS=""

export OUTFOLDER="image_files"
mkdir -p "$OUTFOLDER" 
while read line
do
    export THISQUERY="`echo $line | awk '-F\t' '{print $1}'`"
    export URL="`echo $line | awk '-F\t' '{print $2}'`"
    export BASEFILENAME="`basename \"$URL\"`"
    export NOQUERY="`echo -n \"$BASEFILENAME\" | sed 's/\?.*$//'`"
    export MD5NAME="`echo -n $URL | md5 | awk '{print $1}'`"
    export OUTFILE="${MD5NAME}__${NOQUERY}"

    #mkdir -p "images/$THISQUERY" #"headers/$THISQUERY"
    if [ ! -f "$OUTFOLDER/$OUTFILE" ]; then
        echo "Requesting '$URL' to '$OUTFOLDER/$OUTFILE'.">&2
        curl --retry 3 -o "$OUTFOLDER/$OUTFILE" -D - -A "$USERAGENT" -s -m 10 "$URL" >&2
        if [ -f "$OUTFOLDER/$OUTFILE" ]
        then
            echo -e "$THISQUERY\t$URL\t$OUTFILE"
        fi
    fi
    echo

done

