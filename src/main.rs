#[macro_use]
extern crate clap;
extern crate image;

use image::{GenericImageView, GenericImage};

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

        print!("{}x{}: ", w, h);
        let (w, h) = (w/16, h/16);
        println!("{}x{}", w, h);
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

    /// Simple euclidiand distance between two slices
    pub fn distance(&self, other: &Block) -> f64 {
        let mut sum = 0.;
        for i in 0..self.data.len() {
            let a = self.data[i];
            let b = other.data[i];
            // Subtract preventing underflow
            let diff = a.max(b) - a.min(b);
            sum += (diff * diff) as f64;
        }

        sum.sqrt()
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
                assert!(d == &1)
            }
        }
    }
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
    let _mark = image::open(mark).expect("Unable to open watermark").to_luma();

    let blocks = Block::from_image(&mut image);

    println!("{}", output);
}
