#import "ImageRecognizer.h"
#include "URLHelper.h"
#include "TensorFlowInference.h"

#import <ImageIO/ImageIO.h>

#include <string>
#include <fstream>

@implementation ImageRecognizer
{
    TensorFlowInference * inference;
    NSDictionary * labels;
    NSNumber * imageMean;
    NSNumber * imageStd;
}

- (id) initWithData:(NSString *)modelInput labels:(NSString *)labelsInput
{
    self = [super init];
    if (self != nil) {
        TensorFlowInference * tensorFlowInference = [[TensorFlowInference alloc] initWithModel:modelInput];
        inference = tensorFlowInference;
        labels = loadLabels(labelsInput);
    }
    return self;
}

- (NSArray *) recognizeImage:(NSString *)image maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold
{
    NSNumber * maxResultsResolved = maxResults != nil ? maxResults : [NSNumber numberWithInt:3];
    NSNumber * thresholdResolved = threshold != nil ? threshold : [NSNumber numberWithFloat:0.1];

    tensorflow::Tensor tensor = createImageTensor(image);
    [inference feed:@"image_tensor" tensor:tensor];

    NSArray * outputNames = [NSArray arrayWithObjects:@"detection_classes", @"detection_scores", @"detection_boxes", @"num_detections", nil];
    [inference run:outputNames enableStats:false];

    NSArray * num_output = [inference fetch:@"num_detections"];
    if ([num_output count] != 1) {
        throw std::invalid_argument("wrong number of detections");
    }
    int num = [[num_output objectAtIndex:0] intValue];

    NSArray * classes_output = [inference fetch:@"detection_classes"];
    if ([classes_output count] != num) {
        throw std::invalid_argument("wrong number of detection classes");
    }
    NSArray * scores_output = [inference fetch:@"detection_scores"];
    if ([scores_output count] != num) {
        throw std::invalid_argument("wrong number of detection scores");
    }
    NSArray * boxes_output = [inference fetch:@"detection_boxes"];
    if ([boxes_output count] != num * 4) {
        throw std::invalid_argument("wrong number of detection boxes");
    }

    NSMutableArray * results = [NSMutableArray new];
    for (int i = 0; i < [scores_output count]; i++) {
        NSNumber * score = [scores_output objectAtIndex:i];
        if ([score floatValue] >= [thresholdResolved floatValue]) {
            NSMutableDictionary * entry = [NSMutableDictionary dictionary];
            NSNumber * item_id = [classes_output objectAtIndex:i];
            entry[@"score"] = score;
            entry[@"item_id"] = item_id;
            entry[@"name"] = labels[item_id][@"name"];
            entry[@"display_name"] = labels[item_id][@"display_name"];
            entry[@"box"] = [boxes_output subarrayWithRange: NSMakeRange(i * 4, 4)];
            [results addObject:entry];
        }
    }

    NSArray * resultsSorted = [results sortedArrayUsingComparator:^NSComparisonResult(id first, id second) {
      return [second[@"score"] compare:first[@"score"]];
    }];
    auto finalSize = MIN([resultsSorted count], [maxResultsResolved integerValue]);
    return [resultsSorted subarrayWithRange:NSMakeRange(0, finalSize)];
}

tensorflow::Tensor createImageTensor(NSString * imageUri) {
    NSURL * url = [URLHelper toURL:imageUri];

    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = imageAsVector(url, &image_width, &image_height, &image_channels);

    const int wanted_channels = 3;
    tensorflow::Tensor image_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, image_height, image_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<unsigned char, 4>();
    tensorflow::uint8* in = image_data.data();

    unsigned char * out = image_tensor_mapped.data();
    for (int y = 0; y < image_height; ++y) {
        tensorflow::uint8* in_row = in + (y * image_width * image_channels);
        unsigned char * out_row = out + (y * image_width * wanted_channels);
        for (int x = 0; x < image_width; ++x) {
            tensorflow::uint8* in_pixel = in_row + (x * image_channels);
            unsigned char * out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = in_pixel[c];
            }
        }
    }

    return image_tensor;
}

std::vector<tensorflow::uint8> imageAsVector(NSURL * imageUrl, int* out_width, int* out_height, int* out_channels) {
    CGImageSourceRef source = CGImageSourceCreateWithURL((CFURLRef)imageUrl, NULL);
    if(source==NULL) {
        throw std::invalid_argument("Failed to create image source from url");
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    if(image==NULL) {
        throw std::invalid_argument("Failed to create image ref from source");
    }

    int orientation = 1;
    CFDictionaryRef dict = CGImageSourceCopyPropertiesAtIndex(source, 0, NULL);
    if(dict)
    {
        CFNumberRef imageOrientation;
        if(CFDictionaryGetValueIfPresent(dict, kCGImagePropertyOrientation, (const void **)&imageOrientation)) {
            if(imageOrientation)
                CFNumberGetValue(imageOrientation, kCFNumberIntType, &orientation);
        }
        CFRelease(dict);
    }
    LOG(INFO) << "Image orientation " << orientation;
    CFRelease(source);

    int width = (int)CGImageGetWidth(image);
    int height = (int)CGImageGetHeight(image);
    int canvasw, canvash;

    if(orientation<=4)
    {
        canvasw = width;
        canvash = height;
    }
    else
    {
        canvasw = height;
        canvash = width;
    }

    const int channels = 4;
    const int bytes_per_row = (canvasw * channels);
    const int bytes_in_image = (bytes_per_row * canvash);
    std::vector<tensorflow::uint8> result(bytes_in_image);
    const int bits_per_component = 8;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        result.data(), canvasw, canvash,
        bits_per_component, bytes_per_row, color_space,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);

    switch(orientation)
    {
        case 2:
            // 2 = 0th row is at the top, and 0th column is on the right - Flip Horizontal
            CGContextConcatCTM(context, CGAffineTransformMake(-1.0, 0.0, 0.0, 1.0, width, 0.0));
            break;

        case 3:
            // 3 = 0th row is at the bottom, and 0th column is on the right - Rotate 180 degrees
            CGContextConcatCTM(context, CGAffineTransformMake(-1.0, 0.0, 0.0, -1.0, width, height));
            break;

        case 4:
            // 4 = 0th row is at the bottom, and 0th column is on the left - Flip Vertical
            CGContextConcatCTM(context, CGAffineTransformMake(1.0, 0.0, 0, -1.0, 0.0, height));
            break;

        case 5:
            // 5 = 0th row is on the left, and 0th column is the top - Rotate -90 degrees and Flip Vertical
            CGContextConcatCTM(context, CGAffineTransformMake(0.0, -1.0, -1.0, 0.0, height, width));
            break;

        case 6:
            // 6 = 0th row is on the right, and 0th column is the top - Rotate 90 degrees
            CGContextConcatCTM(context, CGAffineTransformMake(0.0, -1.0, 1.0, 0.0, 0.0, width));
            break;

        case 7:
            // 7 = 0th row is on the right, and 0th column is the bottom - Rotate 90 degrees and Flip Vertical
            CGContextConcatCTM(context, CGAffineTransformMake(0.0, 1.0, 1.0, 0.0, 0.0, 0.0));
            break;

        case 8:
            // 8 = 0th row is on the left, and 0th column is the bottom - Rotate -90 degrees
            CGContextConcatCTM(context, CGAffineTransformMake(0.0, 1.0, -1.0, 0.0, height, 0.0));
            break;

        default:
            break;
    }

    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGImageRelease(image);

    *out_width = canvasw;
    *out_height = canvash;
    *out_channels = channels;

    return result;
}

NSDictionary * loadLabels(NSString * labelUri) {
    NSData * labelData = [NSData dataWithContentsOfURL:[URLHelper toURL:labelUri]];
    NSString * labelString = [[NSString alloc] initWithData:labelData encoding:NSUTF8StringEncoding];
    NSRegularExpression * regex = [NSRegularExpression
        regularExpressionWithPattern:@"item\\s*\\{[^}]*?name:\\s*\"?([^}]+?)\"?\\s*id:\\s*([^}]+?)\\s*display_name:\\s*\"?([^}]+?)\"?\\s*\\}"
        options:NSRegularExpressionDotMatchesLineSeparators error:nil];
    NSArray * matches = [regex matchesInString:labelString options:0 range:NSMakeRange(0, [labelString length])];
    LOG(INFO) << "Found " << [matches count] << " labels";
    NSMutableDictionary * dict = [NSMutableDictionary dictionary];

    NSNumberFormatter * formatter = [[NSNumberFormatter alloc] init];
    formatter.numberStyle = NSNumberFormatterDecimalStyle;
    for (NSTextCheckingResult * match in matches) {
        NSString * name = [labelString substringWithRange:[match rangeAtIndex:1]];
        NSNumber * item_id = [formatter numberFromString:[labelString substringWithRange:[match rangeAtIndex:2]]];
        NSString * displayName = [labelString substringWithRange:[match rangeAtIndex:3]];
        dict[item_id] = @{
          @"name":[name stringByReplacingOccurrencesOfString:@"\\'" withString:@"'"],
          @"item_id":item_id,
          @"display_name":[displayName stringByReplacingOccurrencesOfString:@"\\'" withString:@"'"] };
    }
    return dict;
}

@end
