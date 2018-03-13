#import "RNVideoRecognition.h"

#include "VideoRecognizer.h"

#import "RCTUtils.h"

#include <string>
#include <fstream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"

@implementation RNVideoRecognition
{
    std::unordered_map<std::string, VideoRecognizer *> videoRecognizers;
}

@synthesize bridge = _bridge;

- (dispatch_queue_t)methodQueue
{
    return dispatch_get_main_queue();
}

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(initVideoRecognizer:(NSString *)tId data:(NSDictionary *)data resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
{
    try {
        NSString * model = data[@"model"];
        NSString * labels = data[@"labels"];

        VideoRecognizer * videoRecognizer = [[VideoRecognizer alloc] initWithData:model labels:labels];
        videoRecognizers[[tId UTF8String]] = videoRecognizer;

        resolve(@1);
    } catch( std::exception& e ) {
        reject(RCTErrorUnspecified, @(e.what()), nil);
    }
}

RCT_EXPORT_METHOD(recognize:(NSString *)tId data:(NSDictionary *)data resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
{
    try {
        NSString * video = [data objectForKey:@"video"];
        NSNumber * maxResults = [data objectForKey:@"maxResults"];
        NSNumber * threshold = [data objectForKey:@"threshold"];

        VideoRecognizer * videoRecognizer = videoRecognizers[[tId UTF8String]];
        NSArray * result = [videoRecognizer recognize:video maxResults:maxResults threshold:threshold];
        resolve(result);
    } catch( std::exception& e ) {
        reject(RCTErrorUnspecified, @(e.what()), nil);
    }
}

RCT_EXPORT_METHOD(close:(NSString *)tId resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
{
    try {
        videoRecognizers.erase([tId UTF8String]);
        resolve(@1);
    } catch( std::exception& e ) {
        reject(RCTErrorUnspecified, @(e.what()), nil);
    }
}

@end

