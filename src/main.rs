#[macro_use]
extern crate clap;
extern crate image;
extern crate rayon;
extern crate rand;

use image::{GenericImageView, GenericImage, ImageBuffer};
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::{File};

mod block;
use block::Block;

mod kmeans;
use kmeans::*;

pub fn blocks_to_image(width: u32, height: u32, blocks: &Vec<Block>) -> (u32, u32, Vec<u8>) {
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
    let mark = image::open(mark).expect("Unable to open watermark").to_luma();

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

    // Use the median to convert the variance to a polarities matrix
    let variance: Vec<u8> = variance
        .par_iter_mut()
        .map(|v| if *v < median {0u8} else {1u8})
        .collect();

    // Create a permutation key and permuted watermark
    let mut rng = thread_rng();
    let (mark_width, mark_height) = mark.dimensions();
    let watermark: Vec<u8> = mark.into_raw().par_iter().map(|x| if *x > 0 {1} else {0}).collect();
    let mut permuted = watermark.clone();

    let mut key_1: Vec<u8> = Vec::with_capacity(watermark.len());
    let mut row_permute = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    for r in 0..mark_height {
        row_permute.shuffle(&mut rng);

        let start = (r*mark_width) as usize;
        let end = ((r+1)*mark_width) as usize;

        let row = &watermark[start..end];
        let mut permuted_row = &mut permuted[start..end];
        for (i, j) in row_permute.iter().map(|i| *i as usize).enumerate() {
            permuted_row[i] = row[j];
        }

        key_1.extend(row_permute.iter());
    }

    let key_2: Vec<u8> = key_1
        .iter()
        .zip(variance.iter())
        .map(|(k, p)| k ^ p)
        .collect();

    let mut extracted = Vec::with_capacity(watermark.len());
    for r in 0..mark_height {

        let start = (r*mark_width) as usize;
        let end = ((r+1)*mark_width) as usize;

        let key = &key_1[start..end];
        let row = &permuted[start..end];
        for j in key.iter().map(|i| *i as usize) {
            extracted.push(if 0 == key_2[j] {0u8} else {255u8});
        }
    }

    write_image(output, mark_width, mark_height, &extracted);
}
