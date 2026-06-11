//! Perceptual colormaps for element-magnitude visualisations.
//!
//! The byte-Hilbert / literal-byte path uses arbvis's
//! [`arbvis::color::build_pixel_lut`] — the Stairwell binary-file forensic
//! scheme, whose *categorical* hue bands (null=black, control=green,
//! ASCII=blue, high=red, 0xFF=white) carry semantic meaning when the byte is
//! a literal file byte.
//!
//! For the MoE summary / CKA heatmaps the byte is instead a *normalised
//! magnitude* (`0..=255`), where the Stairwell scheme is actively misleading:
//! it is non-monotonic in luminance (green reads brighter than blue), so a
//! larger value can look dimmer than a smaller one, and its category edges
//! (e.g. blue→red at 0x7F) paint false contour lines through continuous data.
//! It also reuses the same green/red the signed diff LUT
//! ([`arbvis::color::build_diff_signed_lut`]) uses, so a "red" region means
//! "high magnitude" here but "value shrank" there.
//!
//! [`CIVIDIS_LUT`] is the fix for those views: matplotlib's `cividis`
//! colormap — perceptually uniform, monotonic in luminance (bigger = brighter,
//! always), and colour-vision-deficiency safe (it varies dark-blue → yellow,
//! avoiding the red/green axis entirely). It shares no hue with the diff LUT,
//! so the magnitude view is unmistakable at a glance.
//!
//! The 256 entries are matplotlib's canonical `_cividis_data` (each float
//! channel rounded to `0..=255`); regenerate from
//! `matplotlib/_cm_listed.py` if ever updated.

use image::Rgb;

/// 256-entry cividis colormap (matplotlib `cividis`), dark-blue `(0,34,78)`
/// → yellow `(254,232,56)`. Indexed by a normalised-magnitude byte exactly as
/// the Stairwell LUT is, so it drops into the existing
/// [`crate::layout::render::plain_element_color`] path unchanged.
pub static CIVIDIS_LUT: [Rgb<u8>; 256] = [
    Rgb([0, 34, 78]),
    Rgb([0, 35, 79]),
    Rgb([0, 36, 81]),
    Rgb([0, 37, 83]),
    Rgb([0, 37, 84]),
    Rgb([0, 38, 86]),
    Rgb([0, 39, 88]),
    Rgb([0, 40, 89]),
    Rgb([0, 40, 91]),
    Rgb([0, 41, 93]),
    Rgb([0, 42, 95]),
    Rgb([0, 42, 97]),
    Rgb([0, 43, 98]),
    Rgb([0, 44, 100]),
    Rgb([0, 44, 102]),
    Rgb([0, 45, 104]),
    Rgb([0, 46, 106]),
    Rgb([0, 46, 108]),
    Rgb([0, 47, 109]),
    Rgb([0, 48, 111]),
    Rgb([0, 48, 112]),
    Rgb([0, 49, 112]),
    Rgb([0, 49, 113]),
    Rgb([1, 50, 113]),
    Rgb([5, 51, 113]),
    Rgb([8, 51, 112]),
    Rgb([12, 52, 112]),
    Rgb([15, 53, 112]),
    Rgb([18, 53, 112]),
    Rgb([20, 54, 112]),
    Rgb([22, 55, 112]),
    Rgb([24, 55, 111]),
    Rgb([26, 56, 111]),
    Rgb([28, 57, 111]),
    Rgb([30, 58, 111]),
    Rgb([32, 58, 111]),
    Rgb([33, 59, 110]),
    Rgb([35, 60, 110]),
    Rgb([36, 60, 110]),
    Rgb([38, 61, 110]),
    Rgb([39, 62, 110]),
    Rgb([41, 63, 110]),
    Rgb([42, 63, 109]),
    Rgb([43, 64, 109]),
    Rgb([45, 65, 109]),
    Rgb([46, 65, 109]),
    Rgb([47, 66, 109]),
    Rgb([49, 67, 109]),
    Rgb([50, 67, 109]),
    Rgb([51, 68, 109]),
    Rgb([52, 69, 108]),
    Rgb([53, 69, 108]),
    Rgb([54, 70, 108]),
    Rgb([56, 71, 108]),
    Rgb([57, 72, 108]),
    Rgb([58, 72, 108]),
    Rgb([59, 73, 108]),
    Rgb([60, 74, 108]),
    Rgb([61, 74, 108]),
    Rgb([62, 75, 108]),
    Rgb([63, 76, 108]),
    Rgb([64, 76, 108]),
    Rgb([65, 77, 108]),
    Rgb([66, 78, 108]),
    Rgb([67, 78, 108]),
    Rgb([68, 79, 108]),
    Rgb([69, 80, 108]),
    Rgb([70, 81, 108]),
    Rgb([71, 81, 108]),
    Rgb([72, 82, 108]),
    Rgb([73, 83, 108]),
    Rgb([74, 83, 108]),
    Rgb([75, 84, 108]),
    Rgb([76, 85, 108]),
    Rgb([77, 85, 108]),
    Rgb([78, 86, 108]),
    Rgb([79, 87, 108]),
    Rgb([80, 87, 108]),
    Rgb([81, 88, 109]),
    Rgb([82, 89, 109]),
    Rgb([83, 90, 109]),
    Rgb([84, 90, 109]),
    Rgb([85, 91, 109]),
    Rgb([85, 92, 109]),
    Rgb([86, 92, 109]),
    Rgb([87, 93, 109]),
    Rgb([88, 94, 109]),
    Rgb([89, 94, 110]),
    Rgb([90, 95, 110]),
    Rgb([91, 96, 110]),
    Rgb([92, 97, 110]),
    Rgb([93, 97, 110]),
    Rgb([94, 98, 110]),
    Rgb([94, 99, 111]),
    Rgb([95, 99, 111]),
    Rgb([96, 100, 111]),
    Rgb([97, 101, 111]),
    Rgb([98, 101, 111]),
    Rgb([99, 102, 112]),
    Rgb([100, 103, 112]),
    Rgb([101, 104, 112]),
    Rgb([101, 104, 112]),
    Rgb([102, 105, 112]),
    Rgb([103, 106, 113]),
    Rgb([104, 106, 113]),
    Rgb([105, 107, 113]),
    Rgb([106, 108, 113]),
    Rgb([107, 109, 114]),
    Rgb([108, 109, 114]),
    Rgb([108, 110, 114]),
    Rgb([109, 111, 114]),
    Rgb([110, 111, 115]),
    Rgb([111, 112, 115]),
    Rgb([112, 113, 115]),
    Rgb([113, 114, 116]),
    Rgb([114, 114, 116]),
    Rgb([114, 115, 116]),
    Rgb([115, 116, 117]),
    Rgb([116, 116, 117]),
    Rgb([117, 117, 117]),
    Rgb([118, 118, 118]),
    Rgb([119, 119, 118]),
    Rgb([119, 119, 119]),
    Rgb([120, 120, 119]),
    Rgb([121, 121, 119]),
    Rgb([122, 122, 120]),
    Rgb([123, 122, 120]),
    Rgb([124, 123, 120]),
    Rgb([125, 124, 120]),
    Rgb([126, 124, 120]),
    Rgb([126, 125, 120]),
    Rgb([127, 126, 120]),
    Rgb([128, 127, 120]),
    Rgb([129, 127, 120]),
    Rgb([130, 128, 121]),
    Rgb([131, 129, 121]),
    Rgb([132, 130, 121]),
    Rgb([133, 130, 121]),
    Rgb([134, 131, 121]),
    Rgb([135, 132, 120]),
    Rgb([136, 133, 120]),
    Rgb([137, 133, 120]),
    Rgb([138, 134, 120]),
    Rgb([139, 135, 120]),
    Rgb([140, 136, 120]),
    Rgb([141, 136, 120]),
    Rgb([142, 137, 120]),
    Rgb([143, 138, 120]),
    Rgb([144, 139, 120]),
    Rgb([145, 139, 120]),
    Rgb([146, 140, 120]),
    Rgb([146, 141, 120]),
    Rgb([147, 142, 120]),
    Rgb([148, 142, 119]),
    Rgb([149, 143, 119]),
    Rgb([150, 144, 119]),
    Rgb([151, 145, 119]),
    Rgb([152, 146, 119]),
    Rgb([153, 146, 119]),
    Rgb([154, 147, 118]),
    Rgb([155, 148, 118]),
    Rgb([156, 149, 118]),
    Rgb([157, 149, 118]),
    Rgb([158, 150, 118]),
    Rgb([159, 151, 117]),
    Rgb([160, 152, 117]),
    Rgb([161, 153, 117]),
    Rgb([162, 153, 117]),
    Rgb([163, 154, 116]),
    Rgb([164, 155, 116]),
    Rgb([165, 156, 116]),
    Rgb([166, 156, 116]),
    Rgb([167, 157, 115]),
    Rgb([168, 158, 115]),
    Rgb([169, 159, 115]),
    Rgb([170, 160, 115]),
    Rgb([171, 160, 114]),
    Rgb([172, 161, 114]),
    Rgb([173, 162, 114]),
    Rgb([174, 163, 113]),
    Rgb([175, 164, 113]),
    Rgb([176, 165, 113]),
    Rgb([177, 165, 112]),
    Rgb([179, 166, 112]),
    Rgb([180, 167, 111]),
    Rgb([181, 168, 111]),
    Rgb([182, 169, 111]),
    Rgb([183, 169, 110]),
    Rgb([184, 170, 110]),
    Rgb([185, 171, 109]),
    Rgb([186, 172, 109]),
    Rgb([187, 173, 109]),
    Rgb([188, 174, 108]),
    Rgb([189, 174, 108]),
    Rgb([190, 175, 107]),
    Rgb([191, 176, 107]),
    Rgb([192, 177, 106]),
    Rgb([193, 178, 106]),
    Rgb([194, 179, 105]),
    Rgb([195, 179, 105]),
    Rgb([196, 180, 104]),
    Rgb([197, 181, 104]),
    Rgb([198, 182, 103]),
    Rgb([199, 183, 103]),
    Rgb([200, 184, 102]),
    Rgb([201, 185, 101]),
    Rgb([203, 185, 101]),
    Rgb([204, 186, 100]),
    Rgb([205, 187, 99]),
    Rgb([206, 188, 99]),
    Rgb([207, 189, 98]),
    Rgb([208, 190, 98]),
    Rgb([209, 191, 97]),
    Rgb([210, 192, 96]),
    Rgb([211, 192, 95]),
    Rgb([212, 193, 95]),
    Rgb([213, 194, 94]),
    Rgb([214, 195, 93]),
    Rgb([215, 196, 92]),
    Rgb([217, 197, 92]),
    Rgb([218, 198, 91]),
    Rgb([219, 199, 90]),
    Rgb([220, 200, 89]),
    Rgb([221, 200, 88]),
    Rgb([222, 201, 88]),
    Rgb([223, 202, 87]),
    Rgb([224, 203, 86]),
    Rgb([225, 204, 85]),
    Rgb([226, 205, 84]),
    Rgb([228, 206, 83]),
    Rgb([229, 207, 82]),
    Rgb([230, 208, 81]),
    Rgb([231, 209, 80]),
    Rgb([232, 210, 79]),
    Rgb([233, 211, 78]),
    Rgb([234, 211, 76]),
    Rgb([235, 212, 75]),
    Rgb([237, 213, 74]),
    Rgb([238, 214, 73]),
    Rgb([239, 215, 72]),
    Rgb([240, 216, 70]),
    Rgb([241, 217, 69]),
    Rgb([242, 218, 68]),
    Rgb([243, 219, 66]),
    Rgb([245, 220, 65]),
    Rgb([246, 221, 63]),
    Rgb([247, 222, 62]),
    Rgb([248, 223, 60]),
    Rgb([249, 224, 58]),
    Rgb([251, 225, 56]),
    Rgb([252, 226, 54]),
    Rgb([253, 227, 52]),
    Rgb([254, 228, 52]),
    Rgb([254, 229, 53]),
    Rgb([254, 230, 54]),
    Rgb([254, 232, 56]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_256_entries_with_expected_endpoints() {
        assert_eq!(CIVIDIS_LUT.len(), 256);
        // Dark blue at the low end, yellow at the high end.
        assert_eq!(CIVIDIS_LUT[0], Rgb([0, 34, 78]));
        assert_eq!(CIVIDIS_LUT[255], Rgb([254, 232, 56]));
    }

    #[test]
    fn luminance_is_monotonic_non_decreasing() {
        // The whole point of switching off the Stairwell scheme: a bigger
        // byte must never look dimmer. Rec.709 luma, integer math.
        let luma = |c: &Rgb<u8>| 2126 * c.0[0] as u32 + 7152 * c.0[1] as u32 + 722 * c.0[2] as u32;
        let mut prev = luma(&CIVIDIS_LUT[0]);
        for entry in &CIVIDIS_LUT[1..] {
            let cur = luma(entry);
            assert!(
                cur >= prev,
                "cividis luminance dipped: {cur} < {prev} — non-monotonic ramp",
            );
            prev = cur;
        }
    }

    #[test]
    fn avoids_pure_red_and_pure_green() {
        // CVD-safety sanity: no entry sits on a saturated red or green that
        // would collide with the signed diff LUT's poles.
        for c in &CIVIDIS_LUT {
            let [r, g, b] = c.0;
            let pure_red = r > 100 && g < 40 && b < 40;
            let pure_green = g > 100 && r < 40 && b < 40;
            assert!(
                !pure_red && !pure_green,
                "unexpected pole-like color {:?}",
                c.0
            );
        }
    }
}
