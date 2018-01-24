#import "ImageRecognizer.h"

#include "TensorFlowInference.h"

#include <string>
#include <fstream>

@implementation ImageRecognizer
{
    TensorFlowInference * inference;
    NSDictionary * labels;
    NSNumber * imageMean;
    NSNumber * imageStd;
}

- (id) initWithData:(NSString *)modelInput labels:(NSString *)labelsInput imageMean:(NSNumber *)imageMeanInput imageStd:(NSNumber *)imageStdInput
{
    self = [super init];
    if (self != nil) {
        imageMean = imageMeanInput != nil ? imageMeanInput : [NSNumber numberWithInt:117];
        imageStd = imageStdInput != nil ? imageStdInput : [NSNumber numberWithFloat:1];

        TensorFlowInference * tensorFlowInference = [[TensorFlowInference alloc] initWithModel:modelInput];
        inference = tensorFlowInference;
        labels = loadLabels(labelsInput);
    }
    return self;
}

- (NSArray *) recognizeImage:(NSString *)image inputName:(NSString *)inputName inputSize:(NSNumber *)inputSize
         outputNames:(NSArray *)outputNames maxResults:(NSNumber *)maxResults
         thresholdOutputName:(NSString *)thresholdOutputName threshold:(NSNumber *)threshold
         labelOutputName: (NSString *)labelOutputName
{
    NSString * inputNameResolved = inputName != nil ? inputName : @"input";
    NSNumber * inputSizeResolved = inputSize != nil ? inputSize : [NSNumber numberWithInt:224];
    NSNumber * maxResultsResolved = maxResults != nil ? maxResults : [NSNumber numberWithInt:3];
    NSNumber * thresholdResolved = threshold != nil ? threshold : [NSNumber numberWithFloat:0.1];

    NSData * imageData = loadFile(image);

    tensorflow::Tensor tensor = createImageTensor(imageData, "jpg", [inputSizeResolved floatValue], [imageMean floatValue], [imageStd floatValue]);
    [inference feed:inputNameResolved tensor:tensor];
    [inference run:outputNames enableStats:false];
    
    NSMutableDictionary * dict = [NSMutableDictionary dictionary];
    for (NSString * outputName in outputNames) {
        dict[outputName] = [inference fetch:outputName];
    }

    NSMutableArray * results = [NSMutableArray new];
    NSArray * thresholdOutput = dict[thresholdOutputName];
    for (int i = 0; i < [thresholdOutput count]; i++) {
        NSNumber * score = [thresholdOutput objectAtIndex:i];
        if ([score floatValue] >= [thresholdResolved floatValue]) {
            NSMutableDictionary * entry = [NSMutableDictionary dictionary];
            for (NSString * outputName in [dict allKeys]) {
                id output = [dict[outputName] objectAtIndex:i];
                entry[outputName] = output;
                if (labelOutputName && [outputName isEqualToString:labelOutputName]) {
                    entry[@"name"] = labels[output][@"name"];
                    entry[@"display_name"] = labels[output][@"display_name"];
                }
            }
            [results addObject:entry];
        }
    }

    NSArray * resultsSorted = [results sortedArrayUsingComparator:^NSComparisonResult(id first, id second) {
      return [second[thresholdOutputName] compare:first[thresholdOutputName]];
    }];
    auto finalSize = MIN([resultsSorted count], [maxResultsResolved integerValue]);
    return [resultsSorted subarrayWithRange:NSMakeRange(0, finalSize)];
}

tensorflow::Tensor createImageTensor(NSData * data, const char* image_type, float input_size, float input_mean, float input_std) {
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = imageAsVector(data, image_type, &image_width, &image_height, &image_channels);

    const int wanted_width = input_size;
    const int wanted_height = input_size;
    const int wanted_channels = 3;

    tensorflow::Tensor image_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<unsigned char, 4>();
    tensorflow::uint8* in = image_data.data();

    unsigned char * out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        unsigned char * out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            unsigned char * out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }

    return image_tensor;
}

std::vector<tensorflow::uint8> imageAsVector(NSData * data, const char* image_type, int* out_width, int* out_height, int* out_channels) {

    CFDataRef file_data_ref =  (__bridge CFDataRef)data;
    CGDataProviderRef image_provider = CGDataProviderCreateWithCFData(file_data_ref);

    CGImageRef image;
    if (strcasecmp(image_type, "png") == 0) {
        image = CGImageCreateWithPNGDataProvider(image_provider, NULL, true,
                                                 kCGRenderingIntentDefault);
    } else {
        try {
            image = CGImageCreateWithJPEGDataProvider(image_provider, NULL, true,
                                                      kCGRenderingIntentDefault);
        } catch( std::exception& e ) {
            CFRelease(image_provider);
            CFRelease(file_data_ref);
            fprintf(stderr, "Unknown image type\n");
            *out_width = 0;
            *out_height = 0;
            *out_channels = 0;
            return std::vector<tensorflow::uint8>();
        }
    }

    const int width = (int)CGImageGetWidth(image);
    const int height = (int)CGImageGetHeight(image);
    LOG(INFO) << "Image width: " << width << " height: " << height;
    const int channels = 4;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = (width * channels);
    const int bytes_in_image = (bytes_per_row * height);
    std::vector<tensorflow::uint8> result(bytes_in_image);
    const int bits_per_component = 8;
    CGContextRef context = CGBitmapContextCreate(result.data(), width, height,
                                                 bits_per_component, bytes_per_row, color_space,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CFRelease(image_provider);
    CFRelease(file_data_ref);

    *out_width = width;
    *out_height = height;
    *out_channels = channels;
    return result;
}


NSData* loadFile(NSString * uri) {
    NSURL *url = [NSURL URLWithString:uri];
    if (url && url.scheme && url.host) {
        LOG(INFO) << "Loading URL " << [uri UTF8String];
        return [[NSData alloc] initWithContentsOfURL: url];
    }

    if ([[NSFileManager defaultManager] fileExistsAtPath:uri]) {
        LOG(INFO) << "Loading File " << [uri UTF8String];
        return [[NSData alloc] initWithContentsOfFile:uri];
    }

    LOG(INFO) << "Loading Resource " << [uri UTF8String];
    NSString * path = [[NSBundle mainBundle] pathForResource:[uri stringByDeletingPathExtension] ofType:[uri pathExtension]];
    return [NSData dataWithContentsOfFile:path];
}

NSDictionary * loadLabels(NSString * labelUri) {
    NSData * labelData = loadFile(labelUri);
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
        dict[item_id] = @{ @"name":name, @"item_id":item_id, @"display_name":displayName };
    }
    return dict;
}

@end
