#! /bin/bash

stats_files=$(ls data/*.stats)
if [ -z "$stats_files" ]; then
    echo "No stats files found!"
    exit 1
fi
concatenated_stats=""
for file in $stats_files; do
    file_content=$(cat "$file")
    concatenated_stats="${concatenated_stats}
### ${file}

\`\`\`txt
${file_content}
\`\`\`

"
done
echo "stats_content=${concatenated_stats}"
