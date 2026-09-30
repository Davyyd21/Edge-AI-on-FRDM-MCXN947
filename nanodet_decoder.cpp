#include "nanodet_decoder.h"

#include "fsl_debug_console.h"

#include <math.h>
#include <float.h>
#include <stdint.h>


/*
 * NanoDet uses three feature-map levels. Each point predicts the class
 * scores followed by the four DFL distributions used to recover the box.
 */
static const int kNumClasses = 3;
static const int kRegMax = 7;
static const int kBins = 8;
static const int kOutputStride = 35;


/*
 * These thresholds control which predictions are kept before and after NMS.
 * The box-size limits also remove detections that are unrealistically large
 * for the 96x96 model input.
 */
static const float kScoreThreshold = 0.05f;
static const float kNmsIoUThreshold = 0.60f;
static const float kMaxBoxWidth = 0.90f;
static const float kMaxBoxHeight = 0.90f;


/*
 * Convert a quantized int8 value back to the floating-point value used
 * by the model. The output tensor stores values using scale and zero-point.
 */
static inline float Dequantize(int8_t value,float scale,int32_t zero_point)
{
    return ((float)value - (float)zero_point) * scale;
}


/*
 * Decode one DFL distribution.
 *
 * NanoDet does not directly predict a single distance for each side of the
 * box. Instead, it predicts 8 values representing a probability distribution
 * over possible distances. Softmax converts them into probabilities and the
 * weighted sum gives the final distance.
 */
static float DistributionIntegral(const int8_t *values,float scale,int32_t zero_point)
{
    float logits[kBins];
    float max_logit = -FLT_MAX;

    for (int i = 0; i < kBins; ++i)
    {
        logits[i] = Dequantize(values[i],scale,zero_point);
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
        }
    }
    /*
     * Subtracting the maximum value before expf() keeps the softmax
     * numerically stable, especially when the logits are large.
     */
    float sum = 0.0f;

    for (int i = 0; i < kBins; ++i)
    {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }

    float result = 0.0f;

    if (sum > 0.0f)
    {
        for (int i = 0; i < kBins; ++i)
        {
            result += (logits[i] / sum) * (float)i;
        }
    }

    return result;
}


/*
 * Calculate the Intersection over Union between two bounding boxes.
 *
 * IoU is used by NMS to determine whether two detections are describing
 * essentially the same object.
 */
static float IoU(const NanoDetDetection *a,const NanoDetDetection *b)
{
    float ix1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    float iy1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    float ix2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    float iy2 = (a->y2 < b->y2) ? a->y2 : b->y2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;

    if (iw <= 0.0f || ih <= 0.0f)
    {
        return 0.0f;
    }

    float intersection = iw * ih;

    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

    float union_area =area_a + area_b - intersection;

    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return intersection / union_area;
}


/*
 * Apply class-aware Non-Maximum Suppression.
 *
 * Detections are first sorted by score, then lower-scoring boxes are removed
 * when they overlap too much with a stronger detection of the same class.
 */
static int ApplyNMS(NanoDetDetection *detections,int count)
{
    /*
     * Sort the detections from highest to lowest confidence.
     * The strongest detection is therefore considered first by NMS.
     */
    for (int i = 0; i < count - 1; ++i)
    {
        int best = i;

        for (int j = i + 1; j < count; ++j)
        {
            if (detections[j].score > detections[best].score)
            {
                best = j;
            }
        }

        if (best != i)
        {
            NanoDetDetection temp = detections[i];
            detections[i] = detections[best];
            detections[best] = temp;
        }
    }

    int keep_count = 0;

    /*
     * Keep a detection unless a stronger detection of the same class already
     * covers most of its area.
     */
    for (int i = 0; i < count; ++i)
    {
        bool keep = true;

        for (int j = 0; j < keep_count; ++j)
        {
            if (detections[i].class_id != detections[j].class_id)
            {
                continue;
            }

            if (IoU(&detections[i], &detections[j]) > kNmsIoUThreshold)
            {
                keep = false;
                break;
            }
        }

        if (keep)
        {
            if (keep_count != i)
            {
                detections[keep_count] = detections[i];
            }

            keep_count++;
        }
    }

    return keep_count;
}


/*
 * Decode the raw output tensor produced by NanoDet.
 *
 * The model has 189 points distributed over three feature-map levels:
 *
 *   12x12 -> stride 8
 *    6x6  -> stride 16
 *    3x3  -> stride 32
 *
 * Each point contains 35 values: 3 class scores followed by four groups
 * of 8 values used by DFL to recover the left, top, right and bottom
 * distances of the bounding box.
 */
int NanoDet_Decode(const int8_t *output,float scale,int32_t zero_point,NanoDetDetection *detections,int max_detections)
{
    if (output == NULL || detections == NULL || max_detections <= 0)
    {
        return 0;
    }

    /*
     * Keep the candidates separate from the final detections because NMS
     * can remove a large number of overlapping predictions.
     */
    static NanoDetDetection candidates[NANODET_MAX_DETECTIONS * 4];

    int candidate_count = 0;

    const int grid_sizes[3] = {12, 6, 3};
    const int strides[3] = {8, 16, 32};

    int point_index = 0;

    for (int level = 0; level < 3; ++level)
    {
        const int grid = grid_sizes[level];
        const int stride = strides[level];

        for (int row = 0; row < grid; ++row)
        {
            for (int col = 0; col < grid; ++col)
            {
                if (point_index >= NANODET_NUM_POINTS)
                {
                    break;
                }

                /*
                 * Every point occupies 35 consecutive values in the output
                 * tensor, so this gives us the start of the current point.
                 */
                const int output_offset = point_index * kOutputStride;
                const int8_t *point = &output[output_offset];

                int best_class = -1;
                float best_score = 0.0f;

                /*
                 * Find the class with the highest score for this point.
                 * The output is quantized, so dequantization is performed
                 * before comparing the scores.
                 */
                for (int c = 0; c < kNumClasses; ++c)
                {
                    float score = Dequantize(
                        point[c],
                        scale,
                        zero_point);

                    /* Keep the score in the expected [0, 1] range. */
                    if (score < 0.0f)
                    {
                        score = 0.0f;
                    }

                    if (score > 1.0f)
                    {
                        score = 1.0f;
                    }

                    if (score > best_score)
                    {
                        best_score = score;
                        best_class = c;
                    }
                }

                /*
                 * Ignore weak predictions before doing the more expensive
                 * bounding-box decoding.
                 */
                if (best_score < kScoreThreshold)
                {
                    point_index++;
                    continue;
                }

                /*
                 * The center of each prediction is determined by its grid
                 * position and the stride of the current feature-map level.
                 */
                float cx = ((float)col + 0.5f) * (float)stride;
                float cy = ((float)row + 0.5f) * (float)stride;

                /*
                 * Decode the four sides of the box using the DFL
                 * distributions stored after the class scores.
                 */
                float left = DistributionIntegral(
                    &point[3],
                    scale,
                    zero_point) * (float)stride;

                float top = DistributionIntegral(
                    &point[11],
                    scale,
                    zero_point) * (float)stride;

                float right = DistributionIntegral(
                    &point[19],
                    scale,
                    zero_point) * (float)stride;

                float bottom = DistributionIntegral(
                    &point[27],
                    scale,
                    zero_point) * (float)stride;

                NanoDetDetection det;

                det.x1 = cx - left;
                det.y1 = cy - top;
                det.x2 = cx + right;
                det.y2 = cy + bottom;
                det.score = best_score;
                det.class_id = best_class;

                /*
                 * Keep the decoded coordinates inside the 96x96 model
                 * image. This also protects the later coordinate conversion
                 * used when drawing the boxes on the camera frame.
                 */
                if (det.x1 < 0.0f)
                {
                    det.x1 = 0.0f;
                }

                if (det.y1 < 0.0f)
                {
                    det.y1 = 0.0f;
                }

                if (det.x2 > 96.0f)
                {
                    det.x2 = 96.0f;
                }

                if (det.y2 > 96.0f)
                {
                    det.y2 = 96.0f;
                }

                float width = det.x2 - det.x1;
                float height = det.y2 - det.y1;

                /*
                 * Very large boxes are discarded. In practice this removes
                 * predictions that cover most of the input image and are
                 * unlikely to represent a useful object detection.
                 */
                if (width > 96.0f * kMaxBoxWidth || height > 96.0f * kMaxBoxHeight)
                {
                    point_index++;
                    continue;
                }

                if (det.x2 <= det.x1 || det.y2 <= det.y1)
                {
                    point_index++;
                    continue;
                }

                /*
                 * Keep the detailed information for high-confidence
                 * detections in the serial output. This is useful when
                 * checking the DFL decoding and box coordinates.
                 */
                if (best_score > 0.50f)
                {
                    PRINTF(
                        "HIGH SCORE: point=%d level=%d row=%d col=%d class=%d score=%d\r\n",
                        point_index,
                        level,
                        row,
                        col,
                        best_class,
                        (int)(best_score * 1000.0f));

                    PRINTF(
                        "  center=(%d,%d) stride=%d\r\n",
                        (int)cx,
                        (int)cy,
                        stride);

                    PRINTF(
                        "  DFL LTRB=(%d,%d,%d,%d)\r\n",
                        (int)(left * 100.0f),
                        (int)(top * 100.0f),
                        (int)(right * 100.0f),
                        (int)(bottom * 100.0f));

                    PRINTF(
                        "  BOX=(%d,%d)-(%d,%d)\r\n",
                        (int)det.x1,
                        (int)det.y1,
                        (int)det.x2,
                        (int)det.y2);
                }

                /*
                 * Store the candidate for NMS. The extra candidate space
                 * allows several predictions to be generated before the
                 * final filtering step.
                 */
                if (candidate_count < (int)(sizeof(candidates) / sizeof(candidates[0])))
                {
                    candidates[candidate_count] = det;
                    candidate_count++;
                }

                point_index++;
            }
        }
    }

    /*
     * Remove overlapping detections and keep the strongest prediction
     * for each sufficiently overlapping object.
     */
    int nms_count = ApplyNMS(candidates, candidate_count);

    int result_count = nms_count;

    if (result_count > max_detections)
    {
        result_count = max_detections;
    }

    if (result_count > NANODET_MAX_DETECTIONS)
    {
        result_count = NANODET_MAX_DETECTIONS;
    }

    /*
     * Copy the final detections into the caller's output buffer.
     */
    for (int i = 0; i < result_count; ++i)
    {
        detections[i] = candidates[i];
    }

    return result_count;
}