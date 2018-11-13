#[macro_use]
extern crate clap;
extern crate image;
extern crate rayon;
extern crate rand;

use image::{GenericImageView, GenericImage, ImageBuffer};
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::{File};
use std::hash::{Hash, Hasher};

/// A block of 16x16 grayscale pixel data
#[derive(Clone)]
struct Block {
    pub x: u32,
    pub y: u32,
    pub data: [u8; 256],
}

impl Block {
    pub fn new(x: u32, y: u32, data: [u8; 256]) -> Self {
        Block {x, y, data}
    }

    /// Collect 16x16 blocks of an image as arrays with 256 elements
    pub fn from_image<I>(image: &mut I) -> Vec<Self>
    where
        I: GenericImage<Pixel=image::Luma<u8>>
    {

        let (w, h) = image.dimensions();

        debug_assert!(w.wrapping_rem(16) == 0);
        debug_assert!(h.wrapping_rem(16) == 0);

        let (w, h) = (w/16, h/16);
        let mut blocks = Vec::with_capacity((w*h) as usize);

        for r in (0..h).map(|r| r * 16) {
            for c in (0..w).map(|c| c * 16) {
                let mut data = [0; 256];
                for (x, y, p) in image.sub_image(r, c, 16, 16).pixels() {
                    data[(x+16*y) as usize] = p.data[0];
                }
                blocks.push(Block::new(r, c, data));
            }
        }

        blocks
    }

    /// Create a new block which is the mean of a set of blocks
    pub fn as_mean<'a, I>(blocks: &'a mut I) -> Self
    where
        I: Iterator<Item=&'a Block>
    {
        let mut mean = [0; 256];
        let mut data = [0; 256];

        let mut n: usize = 0;
        for block in blocks {
            for i in 0..mean.len() {
                mean[i] += block.data[i] as usize;
            }
            n += 1;
        }

        if n != 0 {
            for i in 0..data.len() {
                data[i] = (mean[i]/n) as u8;
            }
        }

        Block {x: 0, y: 0, data}
    }

    /// Simple euclidiand distance between two slices
    pub fn distance(&self, other: &Block) -> f64 {
        let mut sum = 0.;
        for i in 0..self.data.len() {
            let a = self.data[i];
            let b = other.data[i];
            // Subtract preventing underflow
            let diff = (a.max(b) - a.min(b)) as f64;
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        if self.x != other.x || self.y != other.y {
            return false;
        }

        for i in 0..self.data.len() {
            if self.data[i] != other.data[i] {
                return false;
            }
        }

        true
    }
}

impl Eq for Block {}

impl Hash for Block {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        for d in 0..self.data.len() {
            self.data[d].hash(state);
        }
    }
}

#[cfg(test)]
mod block_test {
    use super::*;

    #[test]
    fn dist() {
        let a = Block::new(0, 0, [0; 256]);
        let mut data = [0; 256];
        data[0] = 1;
        let b = Block::new(0, 0, data);
        assert!(a.distance(&b) - 1. < 1e-6);
        assert!(b.distance(&a) - 1. < 1e-6);
    }

    // Make sure all of the values in each block come from the image
    #[test]
    fn blocks_from_image() {
        use image::{Luma, ImageBuffer};

        // Test image where every pixel value is 1
        let mut test = ImageBuffer::from_fn(256, 1024, |_, _| Luma([1]));

        let blocks = Block::from_image(&mut test);
        for b in blocks {
            for d in b.data.iter() {
                // If any value is not 1, it did not come from the image,
                // which implies that the implementation at least fills the
                // data buffer with information from the image
                assert!(*d == 1)
            }
        }
    }
}

fn blocks_to_image(width: u32, height: u32, blocks: &Vec<Block>) -> (u32, u32, Vec<u8>) {
    let mut buffer = vec![0; blocks.len()*256];

    for block in blocks.iter() {
        for d in 0..block.data.len() {
            let x = block.x as usize + d/16;
            let y = block.y + d.wrapping_rem(16) as u32;
            buffer[x + (y*width) as usize ] = block.data[d];
        }
    }

    (width, height, buffer)
}

fn write_image<P>(path: P, width: u32, height: u32, buffer: &[u8]) -> std::io::Result<()>
where
    P: AsRef<std::path::Path> + std::fmt::Display,
{
    use image::{jpeg, ColorType};
    println!("create {}", path);
    let mut file = File::create(path).unwrap();
    jpeg::JPEGEncoder::new(&mut file)
        .encode(buffer, width, height, ColorType::Gray(8))?;

    Ok(())
}

fn kmeans(k: usize, tol: f64, blocks: &Vec<Block>)
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

fn main() {
    let matches = clap_app!(eigenfaces =>
        (version: crate_version!())
        (author: crate_authors!())
        (about: crate_description!())
        (@arg IMAGE: +required "Path to image to watermark")
        (@arg WATERMARK: +required "Path to the watermark to apply")
        (@arg OUTPUT: -o --output +takes_value "Name and path of the watermarked image")
    ).get_matches();

    let image = matches.value_of("IMAGE").unwrap();
    let mark = matches.value_of("WATERMARK").unwrap();
    let output = matches.value_of("OUTPUT").unwrap_or("watermarked.png");

    let mut image = image::open(image).expect("Unable to open image").to_luma();
    let (width, height) = image.dimensions();
    let _mark = image::open(mark).expect("Unable to open watermark").to_luma();

    let blocks = Block::from_image(&mut image);

    let means = kmeans(20, 1e-6, &blocks);

    let mut approximate = blocks.clone();
    for (centroid, block_ids) in means.clone() {
        for id in block_ids {
            let (x, y) = (approximate[id].x, approximate[id].y);
            approximate[id] = centroid.clone();
            approximate[id].x = x;
            approximate[id].y = y;
        }
    }
    let (_, _, mut approx_image) = blocks_to_image(width, height, &approximate);
    write_image("img/approximate.jpg", width, height, &approx_image).unwrap();
    approx_image
        .par_iter_mut()
        .zip(image.par_iter_mut())
        .for_each(|(a, i)| *a  = *a.max(i) - *a.min(i));
    write_image("img/difference.jpg", width, height, &approx_image).unwrap();

    let mut difference = ImageBuffer::from_raw(width, height, approx_image).unwrap();
    let means_2 = kmeans(20, 1e-6, &Block::from_image(&mut difference));
    let mut difference = approximate.clone();
    for (centroid, block_ids) in means_2 {
        for id in block_ids {
            let (x, y) = (difference[id].x, difference[id].y);
            difference[id] = centroid.clone();
            difference[id].x = x;
            difference[id].y = y;
        }
    }
    let (_, _, difference_image) = blocks_to_image(width, height, &difference);
    write_image("img/approximate_difference.jpg", width, height, &difference_image).unwrap();

    let codebook = blocks;
    let mut indecies = vec![0; codebook.len()];
    for (k, block_ids) in means.iter().map(|(_, ks)| ks).enumerate() {
        for &id in block_ids.iter() {
            indecies[id] = k;
        }
    }

    fn lindex(w: usize, x: usize, y: usize) -> usize {
        x + y*w
    }
    fn sub(a: usize, b: usize) -> usize {
        a.max(b) - a.min(b)
    }

    let code_width = (indecies.len() as f64).sqrt() as usize;
    let mut variance = vec![0.; indecies.len()];
    for x in 0..code_width {
        for y in 0..code_width {
            // Calculate variance at local neighborhood of indecies ignoring indecies outside
            // the bounds of the codes
            let mut is = [0; 9];
            let mut n = 0.;
            for xx in x.max(1)-1..(x+1.min(code_width)) {
                for yy in y.max(1)-1..(y+1.min(code_width)) {
                    let j = lindex(code_width, xx, yy);
                    is[lindex(3, sub(xx,x), sub(yy,y))] = indecies[j];
                }
                n += 1.;
            }
            let mean = &is.iter().fold(0.,|s, &i| s + i as f64 ) / n;
            let i = lindex(code_width, x, y);
            variance[i] = is.iter().fold(0., |s, &i| s + (i as f64 - mean).powi(2)) / n;
        }
    }
    let mut median = variance.clone();
    fn greater(a: f64, b: f64) -> bool {
        match (a, b) {
            (x, y) if x.is_nan() || y.is_nan() => panic!("NaN found in sort"),
            (_, _) => a < b,
        }
    }

    for e in 1..median.len() {
        let mut j = e;
        while j > 0 && greater(median[j - 1], median[j]) {
            let temp = median[j];
            median[j] = median[j-1];
            median[j-1] = temp;
            j -= 1;
        }
    }
    let median = median[median.len()/2];

    // Use the median to convert the variance to a spectral matrix
    variance
        .par_iter_mut()
        .for_each(|v| *v = if *v < median {0.} else {1.});

    println!("{}", output);
}
