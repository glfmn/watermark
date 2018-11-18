use rand::prelude::*;
use rayon::prelude::*;
use block::*;

pub fn kmeans(k: usize, tol: f64, blocks: &Vec<Block>)
    -> std::collections::HashMap<Block, std::vec::Vec<usize>>
{
    use std::collections::HashMap;

    let mut centroids: Vec<Block> = Vec::with_capacity(k);
    let n = blocks.len();

    // Get k data blocks as initial centroids
    let mut m = k;
    for i in 0..n {
        // select m of remaining n-i
        if (random::<usize>() % (n-i)) < m {
            centroids.push(blocks[i].clone());
            m -= 1;
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

    let mut err = std::f64::MAX;
    while err > tol {
        // Concurrently calculate the closest centroid to each block
        let best: Vec<usize> = blocks.par_iter()
            .map(|ref b| {
                centroids.iter()
                    .map(|ref c| b.distance(c))
                    .enumerate()
                    .fold((0, std::f64::MAX), |acc, (i, d)| {
                        if d < acc.1 {
                            (i, d)
                        } else {
                            acc
                        }
                    })
            })
            .map(|(c, _)| c)
            .collect();

        clusters = HashMap::new();
        for k in 0..centroids.len() {
            clusters.insert(k, Vec::new());
        }
        for (block, &centroid) in best.iter().enumerate() {
            let c = clusters.entry(centroid).or_insert(Vec::new());
            c.push(block);
        }

        // Concurrently calculate means
        let new: Vec<Block> = centroids.par_iter()
            .enumerate()
            .map(|(k, _)| {
                Block::as_mean(&mut clusters[&k].iter().map(|&b| &blocks[b]))
            })
            .collect();

        // Error as sum of the distance between the new and old centroids
        err = centroids.iter()
            .zip(new.iter())
            .fold(0., |sum, (o, n)| sum + n.distance(o));
        centroids = new;
    }

    let mut k_means = HashMap::new();
    for (k, centroid) in centroids.iter().enumerate() {
        k_means.insert(centroid.clone(), clusters[&k].clone());
    }

    k_means
}
