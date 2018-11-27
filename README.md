# Watermarking

Watermark images using k-means to embedd the watermark.

## Contributing

Set up and use the `pre-commit` hook which automates testing each time you commit with:

```
$ ln -s ../../.pre-commit.sh .git/hooks/pre-commit
```

To run the watermark extraction, use:

```
$ cargo run --release -- img/TrumanEatsLunchII.jpg img/smiley.jpg
```

This will extract `img/smiley.jpg` from `img/TrumanEatsLunchII.jpg`.
