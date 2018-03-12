#import "ImageRecognizer.h"
#import "ImageProcessor.h"
#import "URLHelper.h"

#import <ImageIO/ImageIO.h>

#include <string>
#include <fstream>

@implementation ImageRecognizer
{
    ImageProcessor * imageProcessor;
}

- (id) initWithData:(NSString *)modelUri labels:(NSString *)labelUri
{
    self = [super init];
    if (self != nil) {
        imageProcessor = [[ImageProcessor alloc] initWithData:modelUri labels:labelUri];
    }
    return self;
}

- (NSArray *) recognizeImage:(NSString *)imageUri maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold
{
    NSURL * imageUrl = [URLHelper toURL:imageUri];

    CGImageSourceRef source = CGImageSourceCreateWithURL((CFURLRef)imageUrl, NULL);
    if(source==NULL) {
        throw std::invalid_argument("Failed to create image source from url");
    }

    CGImagePropertyOrientation orientation = kCGImagePropertyOrientationUp;
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
    CFRelease(source);

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    if(image==NULL) {
        throw std::invalid_argument("Failed to create image ref from source");
    }

    NSArray * result = [imageProcessor recognize:image orientation:orientation maxResults:maxResults threshold:threshold];

    CGImageRelease(image);
    return result;
}

@end
